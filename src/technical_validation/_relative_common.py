#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared helpers for the household-relative validation (technical validation 6-7).

Concordance / permutation / Wilson / implied-precision helpers are COPIED-ADAPTED
(not imported) from `src/executor_validation/3_concordance_report.py` and
`2_attach_census_spouse.py` -- cross-package import of digit-prefixed modules is not
clean (repo precedent: `executor_validation/_ocr_common.py`); keep the copies in sync
manually. New here: clean-name parsing (`split_person_name`), role-word gender
classification and rep/role alignment, the surname gate, the pointer-first /
relate-fallback household relationship classifier, and a sex-stratified permutation
null that honors the role-gender gating.

Relationship semantics: `classify_relationship` returns the relationship of a census
household member TO the matched inventor, using the IPUMS family-interrelationship
pointers (`sploc`, `momloc`, `poploc` -- pernum pointers within the household) when
present, with a conservative `relate`-code fallback that is valid only via the
household head. Sex refinement (son/brother/father = male, spouse comparator = female)
is the CALLER's job, so composition stats can see the unrefined set.
"""

import json
import math
import os
import re
import time

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

# ---------------------------------------------------------------------------
# copied-adapted from executor_validation (keep in sync manually)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z]+")


def _lap(label, t0):
    now = time.perf_counter()
    print(f"[timing] {label}: {now - t0:.1f}s", flush=True)
    return now


def first_token(name):
    """First uppercased alpha token of a name (the given name), or '' ."""
    m = _TOKEN_RE.search(name or "")
    return m.group(0).upper() if m else ""


def norm_surname(name):
    """Uppercased, whitespace-trimmed surname (kept whole -- surnames may be
    multi-token, e.g. 'VAN DYKE', so don't run them through first_token)."""
    return (name or "").strip().upper()


def load_nicknames(path, tag="[rel]"):
    """Read a nickname-cluster JSON (`{"clusters": [[form, form, ...], ...]}`) and
    return a flat variant->canonical map (all uppercased; canonical = first form of the
    cluster). Missing/empty file -> {} (fuzzy-only concordance, a graceful no-op)."""
    if not path or not os.path.exists(path):
        if path:
            print(f"{tag} WARN nickname file not found ({path}); fuzzy-only "
                  f"concordance.", flush=True)
        return {}
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    clusters = blob.get("clusters", blob) if isinstance(blob, dict) else blob
    nick = {}
    for cluster in clusters:
        if not cluster:
            continue
        canon = str(cluster[0]).strip().upper()
        for form in cluster:
            nick[str(form).strip().upper()] = canon
    return nick


def load_nickmaps(paths, tag="[rel]"):
    """Merge one or more nickname-cluster JSONs into a single variant->canonical
    map. EARLIER paths win on conflicting variants (pass the production nickname
    map first, the period-diminutives file second). PROVENANCE RULE: only clusters
    attested in independent genealogical sources belong in these files -- never
    pairs harvested from this validation's own discordant rows, which would
    inflate concordance circularly."""
    merged = {}
    for path in paths or []:
        for k, v in load_nicknames(path, tag=tag).items():
            merged.setdefault(k, v)
    return merged


def load_given_sex_ref(path, tag="[rel]"):
    """given -> female_share from the `_given_sex_ref.py` parquet. Missing/empty
    path -> {} (role-word gender is then never corrected)."""
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"{tag} WARN given-sex reference not found ({path}); role-word "
              f"gender used as-is.", flush=True)
        return {}
    import polars as pl
    ref = pl.read_parquet(path)
    return dict(zip(ref["given"].to_list(), ref["female_share"].to_list()))


def correct_role_gender(given, role_gender, fshare, hi=0.9, lo=0.1):
    """Correct a representative's role-word gender when the given name's census
    sex distribution decisively contradicts it (>= hi female share flips M->F,
    <= lo flips F->M). Patent role words are noisy -- OCR'd plurals broadcast
    over mixed-sex representative pairs and the masculine form is sometimes used
    for women (the Eugenia-the-'executor' pattern). Ambiguous or unlisted names
    keep the role-word gender."""
    s = fshare.get(given) if given else None
    if s is None:
        return role_gender
    if s >= hi and role_gender == "M":
        return "F"
    if s <= lo and role_gender == "F":
        return "M"
    return role_gender


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, center - half, center + half)


def concordant(a, b, cutoff, nickmap=None):
    """True when two given names match: same nickname/variant cluster, OR a minor
    spelling difference -- one Levenshtein edit always (LULU~LULA), two edits when
    both names have >= 6 letters (HARRIOTT~HARRIET; JOHN~JANE stays rejected) --
    OR fuzzy-match at/above cutoff. The edit rule exists because the InDel ratio
    punishes short names: a single transcription slip on a 4-letter name scores 75.
    NOTE: deliberately looser than the executor_validation copy (which is
    ratio-only) -- the two add-ons' published numbers are separate."""
    if not a or not b:
        return False
    if nickmap:
        ca = nickmap.get(a.upper(), a.upper())
        cb = nickmap.get(b.upper(), b.upper())
        if ca == cb:
            return True
    dist = Levenshtein.distance(a, b)
    if dist <= 1 or (dist <= 2 and min(len(a), len(b)) >= 6):
        return True
    return fuzz.ratio(a, b) >= cutoff


def implied_precision(p_obs, p_null, s):
    """precision from the mixture p_obs = precision*s + (1-precision)*p_null, clamped [0,1]."""
    if s <= p_null:
        return float("nan")
    return max(0.0, min(1.0, (p_obs - p_null) / (s - p_null)))


# ---------------------------------------------------------------------------
# name parsing (clean LLM-extracted names, not raw OCR)
# ---------------------------------------------------------------------------

SUFFIXES = {"JR", "SR", "JUNIOR", "SENIOR", "II", "III", "IV",
            "ND", "RD", "TH", "ESQ", "ESQUIRE"}
TITLES = {"MR", "MRS", "MISS", "DR", "REV", "COL", "CAPT", "GEN", "HON"}
SURNAME_PARTICLES = {"VAN", "VON", "DE", "DER", "DEL", "DELLA", "DI", "DU",
                     "LA", "LE", "MC", "MAC", "ST", "TEN", "TER", "O"}


def split_person_name(full):
    """Clean 'Given [Middle] Surname[, Jr.]' -> (given_first_token, surname, n_tokens).

    Uppercase alpha tokens; leading title tokens and trailing suffix tokens dropped;
    surname = last remaining token, with an immediately preceding particle attached
    ('TEN EYCK', 'VON PHILP'). A single-token name yields given='' (still
    surname-gateable, but unusable for given-name concordance)."""
    toks = [t.upper() for t in _TOKEN_RE.findall(full or "")]
    while toks and toks[0] in TITLES:
        toks = toks[1:]
    while toks and toks[-1] in SUFFIXES:
        toks = toks[:-1]
    if not toks:
        return "", "", 0
    if len(toks) == 1:
        return "", toks[0], 1
    surname = toks[-1]
    body = toks[:-1]
    if len(body) >= 2 and body[-1] in SURNAME_PARTICLES:
        surname = f"{body[-1]} {surname}"
        body = body[:-1]
    given = body[0] if body else ""
    return given, surname, len(toks)


# ---------------------------------------------------------------------------
# role words -> gender gate
# ---------------------------------------------------------------------------

def classify_role_gender(role):
    """'F' | 'M' | 'U' from a role word/phrase. Feminine forms checked FIRST so mixed
    strings like 'administratrix (widow)' resolve F; 'executor/administrator' has no
    feminine marker and resolves M; anything else (trustee, guardian, representative,
    trust officer, ...) is 'U' -- ungated."""
    r = (role or "").strip().lower()
    if "trix" in r or "widow" in r:
        return "F"
    if "trator" in r or "xecutor" in r:
        return "M"
    return "U"


def assign_rep_roles(reps, roles_str):
    """Align semicolon-split role tokens to the semicolon-split representatives.

    len(roles) == len(reps): positional (the aligned case). One role: broadcast.
    Otherwise (rare count-mismatch rows): if every role token classifies to the same
    gender, every rep gets that gender; else every rep gets 'U' (ungated).
    Returns a list of (rep_full, role_str, gender)."""
    roles = [r.strip() for r in (roles_str or "").split(";")]
    roles = [r for r in roles if r]
    if len(roles) == len(reps):
        return [(rep, role, classify_role_gender(role))
                for rep, role in zip(reps, roles)]
    if len(roles) == 1:
        g = classify_role_gender(roles[0])
        return [(rep, roles[0], g) for rep in reps]
    genders = {classify_role_gender(r) for r in roles} or {"U"}
    g = genders.pop() if len(genders) == 1 else "U"
    joined = "; ".join(roles)
    return [(rep, joined, g) for rep in reps]


# ---------------------------------------------------------------------------
# surname gate
# ---------------------------------------------------------------------------

def surname_gate(rep_surname, deceased_surname, cutoff=90.0):
    """('exact'|'fuzzy'|None, score) for a representative surname vs the deceased
    inventor's. Exact after normalization, else rapidfuzz ratio >= cutoff. Both sides
    are clean LLM extractions from the same document, so true relatives should agree
    exactly; the fuzzy fallback (default 90 -- deliberately stricter than the
    OCR-side 80 in golden_negatives/2a) only absorbs residual one-edit slippage."""
    a = norm_surname(rep_surname)
    b = norm_surname(deceased_surname)
    if not a or not b:
        return None, 0.0
    if a == b:
        return "exact", 100.0
    score = fuzz.ratio(a, b)
    return ("fuzzy", score) if score >= cutoff else (None, score)


# ---------------------------------------------------------------------------
# household relationship classifier (pointers first, relate fallback)
# ---------------------------------------------------------------------------

ELIGIBLE_RELATIONSHIPS = ("spouse", "son", "daughter", "brother", "sister",
                          "father", "mother")
FEMALE_RELATIONSHIPS = frozenset({"daughter", "sister", "mother"})
MALE_RELATIONSHIPS = frozenset({"son", "brother", "father"})

_REFINE = {("child", 1): "son", ("child", 2): "daughter",
           ("parent", 1): "father", ("parent", 2): "mother",
           ("sibling", 1): "brother", ("sibling", 2): "sister"}


def refine_relationship(rel, sex):
    """Classifier output + member sex -> the sex-refined eligible relationship name
    (son/daughter, father/mother, brother/sister); 'spouse' passes through with its
    sex carried separately by the caller. Unknown sex -> None (not a comparator)."""
    if rel == "spouse":
        return "spouse"
    return _REFINE.get((rel, sex))


def _ptr(v):
    """Normalize an IPUMS pernum pointer: 0/null -> None (no pointer)."""
    try:
        iv = int(v)
    except (TypeError, ValueError):
        return None
    return iv if iv > 0 else None


def _general_relate(r):
    """IPUMS detailed relate (101, 201, 301, ...) -> general 1..; values already
    below 100 are general codes and pass through."""
    try:
        ir = int(r)
    except (TypeError, ValueError):
        return None
    return ir if ir < 100 else ir // 100


def classify_relationship(inv, m):
    """Relationship of household member `m` TO inventor `inv` (same household,
    m is not inv). Both are dicts with pernum/sploc/momloc/poploc/relate keys
    (values int or None). Returns (relationship, method) with relationship in
    {'spouse','child','parent','sibling'} or (None, None).

    Pointers first (method='pointer'; precedence spouse > child > parent > sibling).
    The relate fallback (method='relate') is used for a relationship ONLY when the
    pointer evidence for that relationship is absent on the relevant side(s) -- it
    never overrides an explicit pointer -- and is valid only via the household head:
    boarder/in-law codes never map."""
    inv_pn, m_pn = _ptr(inv.get("pernum")), _ptr(m.get("pernum"))
    inv_sp, m_sp = _ptr(inv.get("sploc")), _ptr(m.get("sploc"))
    inv_mom, inv_pop = _ptr(inv.get("momloc")), _ptr(inv.get("poploc"))
    m_mom, m_pop = _ptr(m.get("momloc")), _ptr(m.get("poploc"))

    if (inv_sp is not None and inv_sp == m_pn) or (m_sp is not None and m_sp == inv_pn):
        return "spouse", "pointer"
    if (m_mom is not None and m_mom == inv_pn) or (m_pop is not None and m_pop == inv_pn):
        return "child", "pointer"
    if (inv_mom is not None and inv_mom == m_pn) or (inv_pop is not None and inv_pop == m_pn):
        return "parent", "pointer"
    if (inv_mom is not None and inv_mom == m_mom) or \
            (inv_pop is not None and inv_pop == m_pop):
        return "sibling", "pointer"

    gi, gm = _general_relate(inv.get("relate")), _general_relate(m.get("relate"))
    if gi is None or gm is None:
        return None, None
    spouse_ok = inv_sp is None and m_sp is None
    child_ok = m_mom is None and m_pop is None
    parent_ok = inv_mom is None and inv_pop is None
    sibling_ok = inv_mom is None and inv_pop is None and m_mom is None and m_pop is None
    if gi == 1:                                   # inventor is the head
        if gm == 2 and spouse_ok:
            return "spouse", "relate"
        if gm == 3 and child_ok:
            return "child", "relate"
        if gm == 5 and parent_ok:
            return "parent", "relate"
        if gm == 7 and sibling_ok:
            return "sibling", "relate"
    elif gi == 2:                                 # inventor is the head's spouse
        if gm == 1 and spouse_ok:
            return "spouse", "relate"
        if gm == 3 and child_ok:
            return "child", "relate"
    elif gi == 3:                                 # inventor is a child of the head
        if gm in (1, 2) and parent_ok:
            return "parent", "relate"
        if gm == 3 and sibling_ok:
            return "sibling", "relate"
    return None, None


# ---------------------------------------------------------------------------
# sex-stratified permutation null
# ---------------------------------------------------------------------------

class _MatchCache:
    """Memoized `concordant` on (a, b) name pairs -- names repeat heavily."""

    def __init__(self, cutoff, nickmap):
        self.cutoff = cutoff
        self.nickmap = nickmap
        self._cache = {}

    def __call__(self, a, b):
        key = (a, b)
        hit = self._cache.get(key)
        if hit is None:
            hit = concordant(a, b, self.cutoff, self.nickmap)
            self._cache[key] = hit
        return hit

    def any_match(self, names_a, names_b):
        return any(self(a, b) for a in names_a for b in names_b)


GENDER_STRATA = ("F", "M", "U")


def observed_concordance(rows, cutoff, nickmap=None):
    """Boolean flag per row: any gated representative given name matches any
    comparator of the matching stratum (F reps vs F comparators, M vs M, U vs the
    union). `rows` carry `reps_by_gender` / `rels_by_gender` dicts of name lists."""
    cache = _MatchCache(cutoff, nickmap)
    flags = []
    for row in rows:
        hit = any(cache.any_match(row["reps_by_gender"][g], row["rels_by_gender"][g])
                  for g in GENDER_STRATA)
        flags.append(hit)
    return flags


def permutation_null_gated(rows, cutoff, n_perm, seed, nickmap=None):
    """Chance concordance under random re-pairing of representative names across
    rows, RESPECTING the role-gender gate: within each gender stratum g, the rep
    name-lists are permuted among the rows that have reps of that gender, and
    compared against each row's FIXED stratum-g comparators. Without stratification,
    feminine names shuffled onto masculine-role rows would (almost) never match the
    son/brother/father comparators, deflating the null and inflating the lift.

    `rows` must be DETERMINISTICALLY ORDERED by the caller (sort by
    (previous_census_year, PatInvID)) -- one seeded generator drives all strata in
    fixed order, so the result is reproducible. Returns (mean, p2.5, p97.5) over
    `n_perm` replicates."""
    n = len(rows)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    cache = _MatchCache(cutoff, nickmap)
    rng = np.random.default_rng(seed)

    # Precompute, per stratum, which donor NAME-LISTS match which comparator
    # NAME-LISTS. Both repeat heavily (most rows carry a single common given name),
    # so the matrix is built over unique tuples only; replicates then reduce to O(n)
    # set-membership lookups.
    strata = {}
    for g in GENDER_STRATA:
        idx = [i for i, row in enumerate(rows) if row["reps_by_gender"][g]]
        if not idx:
            continue
        donor_uid_of = {}
        donor_uids = [donor_uid_of.setdefault(tuple(rows[i]["reps_by_gender"][g]),
                                              len(donor_uid_of)) for i in idx]
        uid_names = list(donor_uid_of)          # uid -> donor name tuple
        comp_hits = {}                          # comparator tuple -> set of donor uids
        match_uids = []
        for i in idx:
            comps = tuple(rows[i]["rels_by_gender"][g])
            if not comps:
                match_uids.append(frozenset())
                continue
            hits = comp_hits.get(comps)
            if hits is None:
                hits = frozenset(uid for uid, names in enumerate(uid_names)
                                 if cache.any_match(names, comps))
                comp_hits[comps] = hits
            match_uids.append(hits)
        strata[g] = (idx, donor_uids, match_uids)

    rates = np.empty(n_perm, dtype=float)
    for r in range(n_perm):
        null_hit = np.zeros(n, dtype=bool)
        for g in GENDER_STRATA:
            if g not in strata:
                continue
            idx, donor_uids, match_uids = strata[g]
            perm = rng.permutation(len(idx))
            for pos, row_i in enumerate(idx):
                if donor_uids[perm[pos]] in match_uids[pos]:
                    null_hit[row_i] = True
        rates[r] = null_hit.mean()
    return (float(rates.mean()), float(np.percentile(rates, 2.5)),
            float(np.percentile(rates, 97.5)))
