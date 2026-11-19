// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Data.Pdf;

/// <summary>
/// The policy inputs of a path check (RFC 5280, 6.1.1 (c), (e), (f), (g)): the user-initial-policy-set (empty: any policy)
/// and the initial explicit-policy, policy-mapping-inhibit and any-policy-inhibit flags.
/// </summary>
internal readonly record struct CmsPolicyInputs(IReadOnlyList<string> Initial, bool RequireExplicit, bool InhibitMapping, bool InhibitAny)
{
    /// <summary>RFC 5280's defaults: any policy, every flag off (a timestamp authority's path, an indirect list's issuer's).</summary>
    public static CmsPolicyInputs Default => new([], false, false, false);

    /// <summary>The inputs a check's options give a signer's path.</summary>
    public static CmsPolicyInputs From(SignatureCheckOptions options) =>
        new(options.InitialPolicies, options.RequireExplicitPolicy, options.InhibitPolicyMapping, options.InhibitAnyPolicy);

    /// <summary>Whether the initial set is anyPolicy (empty, or holding anyPolicy).</summary>
    public bool AnyInitial => Initial.Count == 0 || Initial.Contains(CmsOids.AnyPolicy);
}

/// <summary>
/// Certificate policy processing on a path (RFC 5280, 6.1.2 to 6.1.5), with the valid policy tree kept as RFC 9618's valid
/// policy graph: the same policies come out, and a node per policy and depth bounds its size by the certificates' policies
/// (a tree can grow exponentially with mappings). The certificate policies, policy mappings, policy constraints and
/// inhibit-any-policy of every certificate below the trusted one are read whether marked critical or not; the trusted
/// certificate's own are not (it is the path's anchor, RFC 5280 6.1). The result is the user-constrained policy set.
/// </summary>
internal static class CmsPolicies
{
    // One node of the graph: a valid policy at a depth, the policies it expects of the next certificate, its parents, and
    // how many children it has (a node left without any is pruned).
    private sealed class Node(string policy, HashSet<string> expected, List<Node> parents)
    {
        public string Policy { get; } = policy;

        public HashSet<string> Expected { get; set; } = expected;

        public List<Node> Parents { get; } = parents;

        public int Children { get; set; }
    }

    /// <summary>
    /// Why the path (first certificate to anchor) fails its policy processing, or null; <paramref name="policies"/> the
    /// user-constrained policy set (ordinal order; anyPolicy when every policy is), empty when it fails or holds none.
    /// </summary>
    public static string? Check(List<CmsCertificate> path, CmsPolicyInputs inputs, out List<string> policies)
    {
        policies = [];
        int n = path.Count - 1;                                                // the certificates below the anchor
        if (n > 0 && !inputs.RequireExplicit && Silent(path))
        {
            // No certificate below the anchor states a policy, a mapping or a requirement: the graph is gone at the first
            // one (6.1.3 (e)), and explicit_policy, from n + 1, cannot reach 0 before the end. Valid, for no policy.
            return null;
        }

        int explicitPolicy = inputs.RequireExplicit ? 0 : n + 1;
        int inhibitAny = inputs.InhibitAny ? 0 : n + 1;
        int policyMapping = inputs.InhibitMapping ? 0 : n + 1;
        var levels = new List<Dictionary<string, Node>>(n + 1)
        {
            new(StringComparer.Ordinal) { [CmsOids.AnyPolicy] = new Node(CmsOids.AnyPolicy, [CmsOids.AnyPolicy], []) },
        };
        bool graph = true;
        for (int i = 1; i <= n; i++)
        {
            var certificate = path[n - i];
            var current = new Dictionary<string, Node>(StringComparer.Ordinal);
            levels.Add(current);
            if (graph && certificate.Policies is { } certified)
            {
                var previous = levels[i - 1];

                // (d) (1): each policy the certificate names, under the nodes expecting it, else under anyPolicy.
                foreach (string policy in certified)
                {
                    if (policy == CmsOids.AnyPolicy)
                    {
                        continue;
                    }

                    var parents = new List<Node>();
                    foreach (var node in previous.Values)
                    {
                        if (node.Expected.Contains(policy))
                        {
                            parents.Add(node);
                        }
                    }

                    if (parents.Count == 0 && previous.TryGetValue(CmsOids.AnyPolicy, out var any))
                    {
                        parents.Add(any);
                    }

                    if (parents.Count > 0)
                    {
                        Add(current, policy, [policy], parents);
                    }
                }

                // (d) (2): anyPolicy, while it is not inhibited (or on a self-issued CA's certificate), stands for every
                // policy the depth above expects that no node names yet.
                if (certified.Contains(CmsOids.AnyPolicy) && (inhibitAny > 0 || (i < n && certificate.SelfIssued)))
                {
                    var expected = new List<string>();
                    foreach (var node in previous.Values)
                    {
                        foreach (string policy in node.Expected)
                        {
                            if (!current.ContainsKey(policy) && !expected.Contains(policy))
                            {
                                expected.Add(policy);
                            }
                        }
                    }

                    foreach (string policy in expected)
                    {
                        var parents = new List<Node>();
                        foreach (var node in previous.Values)
                        {
                            if (node.Expected.Contains(policy))
                            {
                                parents.Add(node);
                            }
                        }

                        Add(current, policy, [policy], parents);
                    }
                }

                // (d) (3): nodes above left without children are pruned.
                graph = Prune(levels, i - 1);
            }
            else
            {
                graph = false;                                                 // (e): no certificate policies
            }

            // (f)
            if (!graph && explicitPolicy == 0)
            {
                return $"{certificate.Named}'s certificate leaves the path no valid certificate policy, and an explicit one is required ({Required(inputs)})";
            }

            if (i == n)
            {
                break;
            }

            // 6.1.4 (a) and (b): policy mappings.
            if (certificate.PolicyMappings is { } mappings)
            {
                foreach (var (issuer, subject) in mappings)
                {
                    if (issuer == CmsOids.AnyPolicy || subject == CmsOids.AnyPolicy)
                    {
                        return $"{certificate.Named}'s certificate maps anyPolicy (RFC 5280 6.1.4 (a): a mapping names policies)";
                    }
                }

                if (graph)
                {
                    foreach (var group in mappings.GroupBy(m => m.Issuer, StringComparer.Ordinal))
                    {
                        var mapped = new HashSet<string>(group.Select(m => m.Subject), StringComparer.Ordinal);
                        if (policyMapping > 0)
                        {
                            if (current.TryGetValue(group.Key, out var node))
                            {
                                node.Expected = mapped;
                            }
                            else if (current.ContainsKey(CmsOids.AnyPolicy) && levels[i - 1].TryGetValue(CmsOids.AnyPolicy, out var any))
                            {
                                Add(current, group.Key, mapped, [any]);
                            }
                        }
                        else if (current.Remove(group.Key, out var removed))
                        {
                            foreach (var parent in removed.Parents)
                            {
                                parent.Children--;
                            }

                            graph = current.Count > 0 && Prune(levels, i - 1);
                            if (!graph)
                            {
                                break;
                            }
                        }
                    }
                }
            }

            // (h), (i), (j): the counters, then the certificate's own limits.
            if (!certificate.SelfIssued)
            {
                explicitPolicy = Math.Max(0, explicitPolicy - 1);
                policyMapping = Math.Max(0, policyMapping - 1);
                inhibitAny = Math.Max(0, inhibitAny - 1);
            }

            if (certificate.RequireExplicitPolicy is { } require && require < explicitPolicy)
            {
                explicitPolicy = require;
            }

            if (certificate.InhibitPolicyMapping is { } inhibit && inhibit < policyMapping)
            {
                policyMapping = inhibit;
            }

            if (certificate.InhibitAnyPolicy is { } skip && skip < inhibitAny)
            {
                inhibitAny = skip;
            }
        }

        // 6.1.5 (a), (b): the last certificate's counter and its own requirement.
        var last = path[0];
        if (n > 0)
        {
            explicitPolicy = Math.Max(0, explicitPolicy - 1);
            if (last.RequireExplicitPolicy == 0)
            {
                explicitPolicy = 0;
            }
        }
        else
        {
            explicitPolicy = Math.Max(0, explicitPolicy - 1);
        }

        // (g), as RFC 9618 5.4 computes it: the authority-constrained set (the valid policies under anyPolicy, and anyPolicy
        // at the last depth), then the user's initial set laid over it.
        var authority = new SortedSet<string>(StringComparer.Ordinal);
        if (graph)
        {
            for (int depth = 1; depth <= n; depth++)
            {
                foreach (var node in levels[depth].Values)
                {
                    if (node.Policy != CmsOids.AnyPolicy && node.Parents.Exists(p => p.Policy == CmsOids.AnyPolicy))
                    {
                        authority.Add(node.Policy);
                    }
                }
            }

            if (levels[n].ContainsKey(CmsOids.AnyPolicy))
            {
                authority.Add(CmsOids.AnyPolicy);
            }
        }

        var user = new SortedSet<string>(authority, StringComparer.Ordinal);
        if (!inputs.AnyInitial)
        {
            user.IntersectWith(inputs.Initial);
            if (authority.Contains(CmsOids.AnyPolicy))
            {
                user.UnionWith(inputs.Initial);
            }
        }

        if (explicitPolicy == 0 && user.Count == 0)
        {
            return inputs.AnyInitial
                ? $"the path is valid for no certificate policy, and an explicit one is required ({Required(inputs)})"
                : $"the path is valid for none of the policies {string.Join(", ", inputs.Initial)} (SignatureCheckOptions.InitialPolicies), and an explicit one is required ({Required(inputs)})";
        }

        policies = [.. user];
        return null;
    }

    // Whether no certificate below the anchor carries a policy extension (policies, mappings, policy constraints).
    private static bool Silent(List<CmsCertificate> path)
    {
        for (int i = 0; i < path.Count - 1; i++)
        {
            if (path[i].Policies is not null || path[i].PolicyMappings is not null || path[i].RequireExplicitPolicy is not null)
            {
                return false;
            }
        }

        return true;
    }

    private static string Required(CmsPolicyInputs inputs) =>
        inputs.RequireExplicit ? $"SignatureCheckOptions.RequireExplicitPolicy, {SignatureCheckOptions.RequireExplicitPolicyVariable}" : "by the path's policy constraints";

    private static void Add(Dictionary<string, Node> level, string policy, HashSet<string> expected, List<Node> parents)
    {
        var node = new Node(policy, expected, parents);
        foreach (var parent in parents)
        {
            parent.Children++;
        }

        level[policy] = node;
    }

    // Prunes the nodes at depth and above that have no children (repeated upward); whether the graph still has a node at
    // the depth below (false: the graph is gone, RFC 5280's NULL tree).
    private static bool Prune(List<Dictionary<string, Node>> levels, int depth)
    {
        for (int d = depth; d >= 0; d--)
        {
            var level = levels[d];
            List<string>? childless = null;
            foreach (var (policy, node) in level)
            {
                if (node.Children == 0)
                {
                    (childless ??= []).Add(policy);
                }
            }

            if (childless is null)
            {
                break;
            }

            foreach (string policy in childless)
            {
                level.Remove(policy, out var node);
                foreach (var parent in node!.Parents)
                {
                    parent.Children--;
                }
            }
        }

        return levels[depth + 1].Count > 0;
    }
}
