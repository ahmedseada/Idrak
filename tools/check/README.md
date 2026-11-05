# tools/check: the pre-build check

Finds every compile error of a project and of every project it references in one pass, before a build:

```
dotnet run -c Release --project tools/check -- tests/Idrak.Tests/Idrak.Tests.csproj
```

The build stops at the first project that fails, so an error in a project that depends on it shows only after that one is
fixed and the build runs again. This tool loads the projects as MSBuild does (the Release configuration; `-c Debug` for
another) and compiles each in memory, in dependency order, with its own language version, nullable context, source
generators and analyzers (and their options), writing nothing. A project whose dependencies have errors is still
checked, against what they declare.

- **Output:** each project's progress as it loads and checks (`[i/n] name: errors, warnings, time`), then every error as
  `path(line,col): error ID: message`, grouped by project, then the counts. A warning the project treats as an error
  (`TreatWarningsAsErrors`) is an error: it fails the build. Other warnings are only counted: they do not fail the build.
- **Exit code:** 0 when there is no error (whatever the warnings), 1 when there are, 2 when a project cannot load.
- **Its compiler:** Roslyn 5.0, the .NET 10 SDK's (`Check.csproj`); keep it the SDK's, so it reports what the build reports.

The fix workflow runs it before the first build and again after fixing errors, before the rebuild (`CLAUDE.md`). It is a
developer tool: not a package, not referenced by the library or the tests (its own `Directory.Build.props` keeps the
repository's package settings and global usings out).
