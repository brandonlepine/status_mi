# Template Families Overview

To isolate identity representations in language models, we construct a diverse set of template families. Each family varies specific linguistic properties while minimizing confounding factors such as behavior, evaluation, or stereotype content. Together, these families enable robust estimation of identity-related directions in activation space.

---

## Family A — Copula Templates

**Purpose:**  
Capture the core identity signal in its most direct form.

**Description:**  
These templates use simple copular constructions (e.g., “is”, “has”) to attach an identity label to a person. They provide the cleanest and most canonical representation of identity as an attribute.

**Examples:**
- “This person is Black.”
- “The person is a Muslim.”
- “This person has Down syndrome.”

**Role in analysis:**  
Serves as the primary source of identity signal. Other families are compared against this baseline to test robustness.

---

## Family B — Person Noun Phrase Templates

**Purpose:**  
Introduce syntactic variation while preserving identity meaning.

**Description:**  
These templates express identity through noun phrases referring to individuals (e.g., “a Black person”, “a Muslim”). This shifts identity from predicate position to noun phrase structure.

**Examples:**
- “This is a Black person.”
- “I met a Muslim.”

**Role in analysis:**  
Tests whether identity representations are invariant to syntactic form (predicate vs noun phrase).

---

## Family C — Semantic Null / Descriptor Templates

**Purpose:**  
Minimize semantic content to isolate identity representations.

**Description:**  
These templates place identity labels in low-information, meta-linguistic contexts (e.g., “The label is X”). They remove behavioral, evaluative, and narrative content.

**Examples:**
- “The demographic descriptor is Black.”
- “The identity mentioned is a Muslim.”

**Role in analysis:**  
Reduces contextual noise and helps ensure that extracted directions reflect identity rather than surrounding semantics.

---

## Family D — Light Natural Context Templates

**Purpose:**  
Test identity representations in minimally realistic language.

**Description:**  
These templates introduce simple, natural contexts (e.g., “I met someone who is X”) without adding behavioral or evaluative content.

**Examples:**
- “I met someone who is Black.”
- “There is someone who has low vision.”

**Role in analysis:**  
Evaluates whether identity directions generalize beyond highly controlled templates.

---

## Family E — Group / Membership Templates

**Purpose:**  
Capture identity as a group-level concept.

**Description:**  
These templates express identity in terms of group membership or collective categories.

**Examples:**
- “The person belongs to the Black community.”
- “These people are Muslims.”

**Role in analysis:**  
Tests whether identity is encoded at the individual level or as part of a broader group representation.

---

## Family F — Fragment Templates

**Purpose:**  
Isolate token-level and phrase-level representations.

**Description:**  
These templates consist of minimal fragments with little to no syntactic structure.

**Examples:**
- “Black person.”
- “A Muslim.”
- “From Nigeria.”

**Role in analysis:**  
Useful for identifying low-level lexical features and for sparse autoencoder (SAE) analyses of activation patterns.

---

## Family G — Plural Templates

**Purpose:**  
Examine identity representations at the population level.

**Description:**  
These templates describe multiple individuals, shifting from singular to plural reference.

**Examples:**
- “These people are Black.”
- “There are people who have cerebral palsy.”

**Role in analysis:**  
Tests whether identity directions change when applied to groups rather than individuals.

---

## Family H — Past Tense Templates

**Purpose:**  
Test temporal invariance of identity representations.

**Description:**  
These templates express identity in past-tense constructions.

**Examples:**
- “This person was Black.”
- “The person had Down syndrome.”

**Role in analysis:**  
Evaluates whether identity encoding depends on temporal framing.

---

## Family I — Future / Modal Templates

**Purpose:**  
Assess identity representations in predictive or hypothetical contexts.

**Description:**  
These templates use future or modal constructions.

**Examples:**
- “This person will be Black.”
- “The person will have low vision.”

**Role in analysis:**  
Tests whether identity representations interact with expectation or prediction structures.

---

## Family J — Position-Shift (Identity-First) Templates

**Purpose:**  
Test invariance to syntactic position and sentence structure.

**Description:**  
These templates place the identity term at the beginning of the sentence or in non-standard syntactic roles.

**Examples:**
- “Black describes the person.”
- “A Muslim is the identity.”

**Role in analysis:**  
Provides a strong test of whether identity is encoded as a stable direction independent of syntactic position.

---

## Summary

Each template family isolates a different dimension of variation:

| Family | Variation Tested |
|--------|----------------|
| A | Core identity signal |
| B | Syntactic structure |
| C | Semantic minimality |
| D | Natural context |
| E | Group representation |
| F | Lexical / fragment-level |
| G | Number (singular vs plural) |
| H | Past temporal framing |
| I | Future / modal framing |
| J | Syntactic position |

Together, these families enable robust estimation of identity directions and help disentangle identity from confounding linguistic factors.