"""The standard security handler, writing side (ISO 32000-2, 7.6.4): revisions 2-4 (RC4 40 to 128 bits, AES-128 with
crypt filters, /EncryptMetadata) and 5-6 (AES-256). Every byte that the specification leaves random (AES vectors, the
revision 5-6 salts and file key, the last 16 bytes of U) comes from a seeded generator, so a file is the same on every
run. RC4 is written here; AES comes from `cryptography`."""
import hashlib
import struct

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from .pdf import N, String

PAD = bytes.fromhex("28BF4E5E4E758A4164004E56FFFA01082E2E00B6D0683E802F0CA9FE6453697A")


class SeededBytes:
    """Deterministic bytes: SHA-256 of a seed, a label and a counter."""

    def __init__(self, *labels):
        self.seed = "|".join(str(x) for x in labels).encode("utf-8")
        self.counter = 0

    def take(self, n):
        out = bytearray()
        while len(out) < n:
            out += hashlib.sha256(self.seed + struct.pack(">Q", self.counter)).digest()
            self.counter += 1
        return bytes(out[:n])


def rc4(key, data):
    s = list(range(256))
    j = 0
    for i in range(256):
        j = (j + s[i] + key[i % len(key)]) & 0xFF
        s[i], s[j] = s[j], s[i]
    out = bytearray(len(data))
    i = j = 0
    for k, b in enumerate(data):
        i = (i + 1) & 0xFF
        j = (j + s[i]) & 0xFF
        s[i], s[j] = s[j], s[i]
        out[k] = b ^ s[(s[i] + s[j]) & 0xFF]
    return bytes(out)


def aes_cbc(key, iv, data, pad=True):
    if pad:
        n = 16 - len(data) % 16
        data = data + bytes([n]) * n
    enc = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
    return enc.update(data) + enc.finalize()


def aes_ecb(key, data):
    enc = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
    return enc.update(data) + enc.finalize()


def pdfdoc_password(pw):
    from .encodings import pdfdoc_encode

    data = pdfdoc_encode(pw)
    if data is None:
        raise ValueError(f"password {pw!r} has characters PDFDocEncoding lacks")
    return data[:32]


def utf8_password(pw):
    import unicodedata

    # SASLprep, as far as these passwords need it: NFKC (no mapped or prohibited characters are used here).
    return unicodedata.normalize("NFKC", pw).encode("utf-8")[:127]


def hash_r6(password, salt, udata):
    """Algorithm 2.B (revision 6)."""
    k = hashlib.sha256(password + salt + udata).digest()
    i = 0
    while True:
        k1 = (password + k + udata) * 64
        e = aes_cbc(k[:16], k[16:32], k1, pad=False)
        s = sum(e[:16]) % 3
        k = (hashlib.sha256, hashlib.sha384, hashlib.sha512)[s](e).digest()
        i += 1
        if i >= 64 and e[-1] <= i - 32:
            return k[:32]


class StandardSecurity:
    """One file's encryption: the /Encrypt dictionary and the string and stream ciphers per object."""

    def __init__(self, revision, user, owner, file_id, permissions=-3904, length=None, cfm=None,
                 encrypt_metadata=True, string_filter="StdCF", seed="idrak"):
        self.revision = revision
        self.user = user
        self.owner = owner
        self.permissions = permissions
        self.encrypt_metadata = encrypt_metadata
        self.string_filter = string_filter
        self.random = SeededBytes(seed, "security", revision, user, owner)
        self.file_id = file_id
        if revision == 2:
            self.v, self.length, self.cfm = 1, 40, "V2"
        elif revision == 3:
            self.v, self.length, self.cfm = 2, length or 128, "V2"
        elif revision == 4:
            self.v, self.length, self.cfm = 4, 128, cfm or "AESV2"
        elif revision in (5, 6):
            self.v, self.length, self.cfm = 5, 256, "AESV3"
        else:
            raise ValueError(revision)
        if revision <= 4:
            self._legacy()
        else:
            self._aes256()

    # -- revisions 2-4 -------------------------------------------------------------------------------------------------

    def _legacy(self):
        n = self.length // 8
        upw = (pdfdoc_password(self.user) + PAD)[:32]
        opw = (pdfdoc_password(self.owner or self.user) + PAD)[:32]
        # Algorithm 3: O.
        k = hashlib.md5(opw).digest()
        if self.revision >= 3:
            for _ in range(50):
                k = hashlib.md5(k[:n]).digest()
        okey = k[:n]
        o = rc4(okey, upw)
        if self.revision >= 3:
            for i in range(1, 20):
                o = rc4(bytes(b ^ i for b in okey), o)
        self.O = o
        # Algorithm 2: the file key.
        h = hashlib.md5()
        h.update(upw)
        h.update(self.O)
        h.update(struct.pack("<i", self.permissions))
        h.update(self.file_id)
        if self.revision >= 4 and not self.encrypt_metadata:
            h.update(b"\xff\xff\xff\xff")
        k = h.digest()
        if self.revision >= 3:
            for _ in range(50):
                k = hashlib.md5(k[:n]).digest()
        self.key = k[:n]
        # Algorithms 4 and 5: U.
        if self.revision == 2:
            self.U = rc4(self.key, PAD)
        else:
            u = rc4(self.key, hashlib.md5(PAD + self.file_id).digest())
            for i in range(1, 20):
                u = rc4(bytes(b ^ i for b in self.key), u)
            self.U = u + self.random.take(16)

    def _object_key(self, num, gen, aes):
        data = self.key + struct.pack("<I", num)[:3] + struct.pack("<I", gen)[:2] + (b"sAlT" if aes else b"")
        return hashlib.md5(data).digest()[:min(len(self.key) + 5, 16)]

    # -- revisions 5-6 -------------------------------------------------------------------------------------------------

    def _aes256(self):
        h = hash_r6 if self.revision == 6 else (lambda pw, salt, u: hashlib.sha256(pw + salt + u).digest())
        upw = utf8_password(self.user)
        opw = utf8_password(self.owner)
        self.key = self.random.take(32)
        uvs, uks, ovs, oks = (self.random.take(8) for _ in range(4))
        self.U = h(upw, uvs, b"") + uvs + uks
        self.UE = aes_cbc(h(upw, uks, b""), bytes(16), self.key, pad=False)
        self.O = h(opw, ovs, self.U) + ovs + oks
        self.OE = aes_cbc(h(opw, oks, self.U), bytes(16), self.key, pad=False)
        perms = struct.pack("<i", self.permissions) + b"\xff\xff\xff\xff" + (b"T" if self.encrypt_metadata else b"F") \
            + b"adb" + self.random.take(4)
        self.Perms = aes_ecb(self.key, perms)

    # -- the ciphers ---------------------------------------------------------------------------------------------------

    def _cipher(self, num, gen, method):
        if method == "Identity":
            return None
        if method == "AESV3":
            key = self.key
        else:
            key = self._object_key(num, gen, method == "AESV2")
        if method == "V2":
            return lambda data: rc4(key, data)
        return lambda data: (lambda iv: iv + aes_cbc(key, iv, data))(self.random.take(16))

    def stream_crypt(self, num, gen):
        return self._cipher(num, gen, self.cfm)

    def string_crypt(self, num, gen):
        return self._cipher(num, gen, "Identity" if self.string_filter == "Identity" else self.cfm)

    def dictionary(self):
        d = {"Filter": N("Standard"), "V": self.v, "R": self.revision, "Length": self.length}
        if self.v >= 4:
            cf = {"CFM": N(self.cfm), "AuthEvent": N("DocOpen"), "Length": self.length // 8 if self.v == 4 else 32}
            d["CF"] = {"StdCF": {"Type": N("CryptFilter"), **cf}}
            d["StmF"] = N("StdCF")
            d["StrF"] = N(self.string_filter)
            if not self.encrypt_metadata:
                d["EncryptMetadata"] = False
        d["O"] = String(self.O, hex=True)
        d["U"] = String(self.U, hex=True)
        if self.revision >= 5:
            d["OE"] = String(self.OE, hex=True)
            d["UE"] = String(self.UE, hex=True)
            d["Perms"] = String(self.Perms, hex=True)
        d["P"] = self.permissions
        return d

    def describe(self):
        return {
            "filter": "Standard", "V": self.v, "R": self.revision, "lengthBits": self.length,
            "cipher": {"V2": "RC4", "AESV2": "AES-128", "AESV3": "AES-256"}[self.cfm],
            "cryptFilter": self.cfm if self.v >= 4 else None,
            "stringFilter": self.string_filter if self.v >= 4 else None,
            "encryptMetadata": self.encrypt_metadata,
            "userPassword": self.user, "ownerPassword": self.owner, "permissions": self.permissions,
        }
