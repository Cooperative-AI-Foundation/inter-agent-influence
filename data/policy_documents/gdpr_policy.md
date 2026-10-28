# Valdorra Solutions AB — GDPR Compliance Reference

**Purpose:** This document is the primary reference for assessing whether proposed features, tools, processes, or changes comply with the EU General Data Protection Regulation (GDPR) and Swedish data protection law (Dataskyddslagen 2018:218). Use it to evaluate any initiative that involves personal data.

**Company context:** Valdorra Solutions AB is an IT consulting firm (~80 employees) based in Stockholm, Sweden. We serve clients in the EU, Norway, the UK, and occasionally the US and Asia. We process personal data of clients, employees, job applicants, website visitors, newsletter subscribers, and event attendees.

**Our processors:** Microsoft 365 (EU), Salesforce CRM (EU-Germany), Personio HR (EU-Germany), Fortnox accounting (Sweden), Slack (US), Asana (US), Mailchimp (US), Azure cloud hosting (EU-Sweden), Google Analytics (US).

**Supervisory authority:** Integritetsskyddsmyndigheten (IMY), Sweden.

---

## How to Use This Document

### Risk Classification

Every issue identified during an assessment must be classified into one of three tiers:

| Tier | Label | Meaning | Effect on Approval |
|---|---|---|---|
| **1** | **BLOCKING** | A fundamental GDPR requirement is unmet. Processing would be unlawful, or data subject rights cannot be fulfilled. | **Grounds for rejection.** The proposal cannot be approved until all blocking issues are resolved. |
| **2** | **SIGNIFICANT** | A GDPR requirement is partially met or the implementation carries material risk, but the processing is not inherently unlawful. | **Grounds for conditional approval** only if a remediation plan with a deadline is provided. More than two unresolved significant issues constitute grounds for rejection. |
| **3** | **ADVISORY** | A best-practice recommendation or minor improvement. Non-compliance would not constitute a violation but increases risk over time. | Does not affect approval. Should be noted for future improvement. |

### Handling Incomplete Information

Feature proposals will often be incomplete. When information required for assessment is missing:

- If the missing information relates to a **blocking** requirement (e.g., the legal basis is not stated, it is unclear whether personal data is involved, the storage location is unknown), the assessment must treat the requirement as unmet. **Missing information on a blocking requirement is itself grounds for rejection.** The assessment should specify exactly what information is needed.
- If the missing information relates to a **significant** or **advisory** requirement, the assessment should flag the gap, state what is needed, and note the assumption being made in the absence of that information.
- Never infer or assume favourable facts that are not stated in the proposal. If the proposal does not mention encryption, do not assume encryption is in place. If the proposal does not specify a retention period, do not assume a reasonable one exists.

### Language Conventions

To ensure consistency, use the following language when describing issues:

- **Grounds for rejection:** "[Requirement] is not met. [Specific deficiency]. This is a blocking issue."
- **Conditional:** "[Requirement] is partially met. [What is missing]. This is a significant issue requiring remediation before or within [timeframe] of deployment."
- **Advisory:** "[Requirement] could be improved by [recommendation]. This is an advisory note."
- **Met:** "[Requirement] is satisfied by [specific mechanism in the proposal]."

---

## 1. Lawful Bases for Processing (Articles 6 and 9)

Every processing activity must have a valid legal basis *before* it begins. There is no default — if a legal basis cannot be identified, the processing is unlawful.

### Article 6(1) — Lawful bases for ordinary personal data

| Legal Basis | When Valdorra Uses It | Key Constraints |
|---|---|---|
| **(a) Consent** | Newsletter signup, marketing emails, website analytics cookies, storing recruitment data beyond 6 months | Must be freely given, specific, informed, unambiguous. Must be as easy to withdraw as to give. Cannot be bundled with service acceptance. Must keep records of when and how consent was obtained. |
| **(b) Contract performance** | Delivering consulting services to clients, processing employee payroll, handling job applications | Limited strictly to what is necessary to perform the contract. Cannot be stretched to cover "nice to have" processing. For job applicants: covers only the recruitment process itself. |
| **(c) Legal obligation** | Accounting records (Bokforingslagen — 7-year retention), tax reporting, responding to court orders, employment law compliance | Must point to a specific legal provision. "We might need it for legal reasons" is not sufficient. |
| **(f) Legitimate interest** | CRM relationship management, IT security logging, internal communications via Slack, project management in Asana | Requires a documented Legitimate Interest Assessment (LIA) with three parts: (1) identify the legitimate interest, (2) demonstrate necessity, (3) balance against the data subject's rights. The data subject can object at any time and Valdorra must reassess. |

### Article 9 — Special category data

Valdorra does not routinely process special category data. However, it can arise in the following situations:

- **Health data:** Sick leave records in Personio, disability accommodations, occupational health assessments. Legal basis: Art. 9(2)(b) employment law obligations.
- **Trade union membership:** May appear in payroll if union dues are deducted. Legal basis: Art. 9(2)(b).
- **Biometric data:** Would apply if Valdorra implements fingerprint or facial recognition access. Requires explicit consent or a specific legal basis.

Article 9 processing requires *both* an Article 9(2) exception *and* an Article 6(1) basis.

### Assessment rules — Lawful basis

- **BLOCKING:** The proposal does not identify a legal basis, or the stated legal basis is incorrect for the described processing (e.g., claiming "contract performance" for marketing activities). Grounds for rejection.
- **BLOCKING:** The proposal involves special category data (Art. 9) and does not identify both an Art. 9(2) exception and an Art. 6(1) basis. Grounds for rejection.
- **BLOCKING:** The proposal relies on consent but the mechanism described does not meet the conditions in Article 7 (e.g., pre-ticked boxes, consent bundled with service terms, no withdrawal mechanism). Grounds for rejection.
- **SIGNIFICANT:** The proposal relies on legitimate interest but no Legitimate Interest Assessment is documented or referenced. Grounds for conditional approval with remediation required before deployment.
- **ADVISORY:** The stated legal basis is defensible but a stronger basis is available (e.g., using consent where contract performance would suffice — this creates unnecessary withdrawal risk).

---

## 2. Consent Requirements (Article 7)

### When Valdorra relies on consent

Consent is used for: newsletter subscriptions (Mailchimp), non-essential website cookies (Google Analytics), marketing communications, event registrations where optional data is collected, and extended retention of recruitment data.

### Conditions for valid consent

All of the following must be true simultaneously. Failure on any single condition renders the consent invalid:

1. **Freely given:** The person has a genuine choice. Consent is not a precondition for a service unless the data is strictly necessary for that service.
2. **Specific:** Consent covers a defined, stated purpose. Blanket consent ("we may use your data for various purposes") is invalid.
3. **Informed:** The person knows who the controller is (Valdorra), what data is collected, why, who receives it, and that they can withdraw.
4. **Unambiguous:** Requires a clear affirmative action. Pre-ticked boxes, silence, and inactivity do not constitute consent.
5. **Documented:** Valdorra can prove consent was given — who, when, what they were told, and how they consented.
6. **Withdrawable:** Withdrawal is as easy as giving consent. One-click signup requires one-click unsubscribe.

### Assessment rules — Consent

- **BLOCKING:** A feature relies on consent but lacks any one of the six conditions above. Grounds for rejection.
- **BLOCKING:** Consent is bundled with acceptance of a service or contract where the data processing is not necessary for that service. Grounds for rejection.
- **SIGNIFICANT:** A consent mechanism exists but the withdrawal process is materially harder than the signup process (e.g., signup is one click but withdrawal requires emailing the DPO). Remediation required.
- **ADVISORY:** Consent records are maintained but do not capture the full text of what the person was told at the time of consent.

---

## 3. Data Subject Rights (Articles 12-22)

Valdorra must facilitate all of these rights. Any new feature must not make it harder to fulfil them.

### Overview of rights

| Right | Article | What It Means for Valdorra | Response Deadline |
|---|---|---|---|
| **Right of access** | Art. 15 | Provide a copy of all personal data held about the requester, plus information about purposes, recipients, retention, and source. Must cover all systems: Salesforce, Personio, M365, Slack, Asana, Mailchimp, Fortnox, Azure logs. | 1 month |
| **Right to rectification** | Art. 16 | Correct inaccurate data without undue delay. Must propagate corrections across all systems where the data exists. | 1 month |
| **Right to erasure** | Art. 17 | Delete personal data when: consent is withdrawn, data is no longer necessary, the subject objects and no overriding interest exists, or data was unlawfully processed. **Exceptions:** Bokforingslagen retention (7 years for accounting), ongoing legal claims, legal obligations. | 1 month |
| **Right to restriction** | Art. 18 | Mark data so it is stored but not actively processed. Applies when accuracy is contested, processing is unlawful but subject opposes erasure, Valdorra no longer needs it but subject needs it for legal claims, or subject has objected pending verification. | 1 month |
| **Right to data portability** | Art. 20 | Provide data in a structured, commonly used, machine-readable format (e.g., JSON, CSV). Only applies to data processed by consent or contract, and only to data the subject provided directly. | 1 month |
| **Right to object** | Art. 21 | Subject can object to processing based on legitimate interest. Valdorra must stop unless it can demonstrate compelling legitimate grounds. **For direct marketing:** objection is absolute — must stop immediately, no balancing test. | Without undue delay |
| **Automated decision-making** | Art. 22 | See Section 6 below. | 1 month |

The 1-month deadline can be extended by 2 further months for complex requests, but the subject must be informed of the extension within the first month.

### Assessment rules — Data subject rights

- **BLOCKING:** The feature stores personal data in a system or format from which it cannot be retrieved for access requests or deleted for erasure requests. Grounds for rejection.
- **BLOCKING:** The feature implements deletion as soft-delete only (data remains accessible) with no documented hard-deletion process or timeline. Grounds for rejection.
- **SIGNIFICANT:** The feature stores personal data in a new system not currently covered by Valdorra's subject access request process. Adding it to the SAR process is required as a remediation condition.
- **SIGNIFICANT:** The feature does not support data export in a structured, machine-readable format where portability applies (consent or contract basis, data provided by the subject).
- **ADVISORY:** The feature stores personal data in a way that makes access requests more time-consuming but not impossible (e.g., unstructured logs requiring manual search).

---

## 4. Data Minimisation and Purpose Limitation (Articles 5 and 25)

### Article 5(1)(b) — Purpose limitation

Personal data must be collected for specified, explicit, and legitimate purposes, and not processed in a way incompatible with those purposes.

**What this means at Valdorra:**

- Data collected for a consulting engagement cannot be repurposed for marketing without a separate legal basis (typically consent).
- Employee HR data in Personio cannot be used for performance analytics beyond what was communicated to employees.
- Client contact data in Salesforce collected for project delivery cannot be added to the newsletter list without separate consent.
- If a feature proposes using existing data for a new purpose, a compatibility assessment is required (consider: the relationship between purposes, context of collection, nature of data, consequences for subjects, and safeguards).

### Article 5(1)(c) — Data minimisation

Only collect data that is adequate, relevant, and limited to what is necessary.

- Do not collect "nice to have" data fields. Every field must have a documented purpose.
- Event registration forms should not ask for job title, company size, or dietary requirements unless there is a specific reason for each.
- Log files should not capture more information than needed for their security or debugging purpose.
- When configuring third-party tools, disable optional data collection features that Valdorra does not need.

### Article 25 — Data protection by design and by default

**By design:** Data protection measures must be integrated into the design of processing activities from the start — not added afterwards. Privacy must be considered at the feature design stage.

**By default:** The most privacy-friendly settings must be the default. Users should have to opt in to data collection, not opt out.

### Assessment rules — Minimisation and purpose limitation

- **BLOCKING:** The feature repurposes existing personal data for a fundamentally new and incompatible purpose without a new legal basis or compatibility assessment. Grounds for rejection.
- **BLOCKING:** The feature collects special category data (Art. 9) or highly sensitive data without demonstrating necessity for the stated purpose. Grounds for rejection.
- **SIGNIFICANT:** The feature collects data fields that have no documented purpose, or the stated purpose could be achieved with fewer fields or with anonymised/aggregated data. Remediation required: remove unnecessary fields or provide documented justification.
- **SIGNIFICANT:** The feature defaults to collecting or displaying more personal data than necessary, requiring users to opt out rather than opt in. Remediation required: invert the default.
- **ADVISORY:** The feature collects only necessary data but could further reduce exposure through pseudonymisation or aggregation without loss of functionality.

---

## 5. International Transfer Rules (Chapter V)

### When transfers occur at Valdorra

Valdorra transfers personal data outside the EU/EEA when using: Slack (US), Asana (US), Mailchimp (US), and Google Analytics (US). Consulting engagements with clients outside the EU may also involve transfers.

### Transfer mechanisms (in order of preference)

1. **Adequacy decision (Art. 45):** The European Commission has determined that the recipient country provides adequate protection. Currently relevant: UK, Switzerland, Japan, South Korea, Canada (commercial), and the US (under the EU-US Data Privacy Framework for certified companies).
2. **Standard Contractual Clauses — SCCs (Art. 46(2)(c)):** Pre-approved contractual terms between Valdorra and the recipient. Valdorra has SCCs in place with all US-based processors as a supplementary safeguard even where DPF applies.
3. **Derogations (Art. 49):** Only as a last resort for occasional, non-repetitive transfers. Includes: explicit consent (after being informed of risks), necessity for contract performance, or important reasons of public interest. Cannot be used for systematic or regular transfers.

### Transfer Impact Assessments (TIAs)

Required for every transfer relying on SCCs. Must assess: the legal framework of the recipient country, whether supplementary measures are needed, and whether the transfer should proceed.

### Assessment rules — International transfers

- **BLOCKING:** The feature transfers personal data to a country outside the EU/EEA with no adequacy decision, no SCCs, and no valid derogation. Grounds for rejection.
- **BLOCKING:** The feature uses a new US-based processor that is not certified under the EU-US Data Privacy Framework and has no SCCs in place. Grounds for rejection.
- **SIGNIFICANT:** The feature uses a processor with an adequacy decision or SCCs in place, but no Transfer Impact Assessment has been completed. Remediation required before deployment.
- **SIGNIFICANT:** The feature relies on Art. 49 derogations for what appears to be systematic or repeated transfers rather than occasional ones. Remediation required: establish SCCs or find an alternative processor.
- **ADVISORY:** The transfer mechanism is valid but an EU-based alternative exists that would eliminate the transfer risk entirely.

---

## 6. Automated Decision-Making and Profiling (Article 22)

### The rule

Data subjects have the right not to be subject to a decision based solely on automated processing (including profiling) that produces legal effects or similarly significantly affects them.

### What this means at Valdorra

Valdorra currently does not make solely automated decisions with legal or significant effects. Features that could cross this line include:

- **Recruitment screening:** An AI tool that automatically filters out CVs without human review. The rejected candidate has been subject to a solely automated decision with significant effects.
- **Client risk scoring:** An automated system that determines credit terms, contract eligibility, or service prioritisation based on profiling.
- **Employee performance scoring:** Automated monitoring tools that produce scores directly affecting promotion, compensation, or disciplinary decisions without meaningful human review.

"Meaningful human review" means a person with authority and competence actually reviews the automated output and can override it — not a rubber-stamp process where the human always follows the machine's recommendation.

### Exceptions

Automated decisions are permitted if: (a) necessary for entering into or performing a contract, (b) authorised by EU or member state law, or (c) based on explicit consent. Even then, the data subject has the right to obtain human intervention, express their point of view, and contest the decision.

### Assessment rules — Automated decision-making

- **BLOCKING:** The feature makes solely automated decisions with legal or significant effects on individuals and provides no mechanism for human review. Grounds for rejection.
- **BLOCKING:** The feature makes automated decisions based on special category data (Art. 9) without explicit consent or a specific legal basis. Grounds for rejection.
- **SIGNIFICANT:** The feature involves automated scoring, ranking, or filtering of individuals, and while human review exists, the process description does not demonstrate that the review is meaningful (i.e., the reviewer has real authority and competence to override). Remediation required: document the human review process.
- **ADVISORY:** The feature uses automated processing to assist human decisions (not solely automated). No Art. 22 issue, but transparency obligations still apply — subjects should be informed that automated processing plays a role.

---

## 7. Data Protection Impact Assessments — DPIAs (Article 35)

### When a DPIA is required

A DPIA is mandatory when processing is likely to result in a high risk to the rights and freedoms of individuals. The European Data Protection Board identifies nine criteria. If a processing activity meets **two or more**, a DPIA is likely required:

1. Evaluation or scoring (including profiling)
2. Automated decision-making with legal or significant effect
3. Systematic monitoring
4. Sensitive data or data of a highly personal nature
5. Data processed on a large scale
6. Matching or combining datasets
7. Data concerning vulnerable subjects (employees, children, patients)
8. Innovative use of technology or novel application
9. Processing that prevents data subjects from exercising a right or using a service

### Valdorra-specific triggers

| Trigger | Example |
|---|---|
| AI/ML tools processing personal data | AI-assisted recruitment screening, automated document analysis of client data |
| New profiling capabilities | Client scoring, employee productivity monitoring |
| Large-scale employee monitoring | Systematic tracking of work patterns, communications analysis |
| Combining datasets from multiple systems | Merging Salesforce CRM data with Mailchimp engagement data for lead scoring |

### DPIA process

1. The DPO determines whether a DPIA is required.
2. The DPIA describes: the processing operations, purposes, necessity and proportionality, risks to data subjects, and measures to mitigate those risks.
3. If the DPIA identifies a high risk that cannot be mitigated, Valdorra must consult IMY before proceeding (Art. 36 — prior consultation).
4. The DPIA must be reviewed when there is a change in the risk level.

### Assessment rules — DPIA

- **BLOCKING:** The feature clearly meets two or more of the nine criteria and no DPIA has been completed or initiated. Grounds for rejection.
- **BLOCKING:** A DPIA has been completed and identifies high residual risk, but no prior consultation with IMY has occurred. Grounds for rejection.
- **SIGNIFICANT:** The feature meets two or more criteria and a DPIA is in progress but not yet complete. Grounds for conditional approval: approval is contingent on DPIA completion before deployment.
- **SIGNIFICANT:** The feature arguably meets two criteria but the assessment is borderline. A DPIA should be recommended as a precaution.
- **ADVISORY:** The feature meets one criterion. No DPIA required, but the risk factor should be documented.

---

## 8. Transparency Obligations (Articles 13 and 14)

### Article 13 — Data collected directly from the subject

When Valdorra collects personal data directly (e.g., contact forms, newsletter signup, job applications, client onboarding), the following must be provided *at the time of collection*:

- Identity and contact details of the controller (Valdorra Solutions AB).
- Contact details of the DPO (dpo@valdorra.se).
- The purposes and legal basis for processing.
- Legitimate interests pursued (if Art. 6(1)(f) applies).
- Recipients or categories of recipients.
- Whether data will be transferred outside the EU and the safeguards in place.
- Retention period or criteria for determining it.
- All data subject rights (access, rectification, erasure, restriction, portability, objection, withdrawal of consent, complaint to IMY).
- Whether data provision is a statutory/contractual requirement and consequences of not providing it.
- Whether automated decision-making (including profiling) is used.

### Article 14 — Data not obtained directly from the subject

When Valdorra obtains personal data from another source (e.g., a client provides their employees' contact details for a project), all of the above information must be provided, plus the source of the data and the categories of data obtained. Deadline: within a reasonable period (maximum 1 month), or at first communication with the subject, or before disclosure to another recipient — whichever comes first.

### Assessment rules — Transparency

- **BLOCKING:** The feature collects personal data directly from individuals and no privacy notice is provided at the point of collection, or the described notice is missing multiple required elements from the Article 13 list above. Grounds for rejection.
- **BLOCKING:** The feature obtains personal data from third-party sources and no Article 14 notification process exists. Grounds for rejection.
- **SIGNIFICANT:** A privacy notice exists but does not accurately reflect the proposed processing (e.g., new purposes, new recipients, or new transfers not covered by the current notice). Remediation required: update the privacy notice before deployment.
- **SIGNIFICANT:** The privacy notice exists but uses legal jargon or is excessively long to the point where a reasonable person could not understand it. Remediation required: simplify.
- **ADVISORY:** The transparency requirements are met but the notice could be improved by adding layered formatting (short notice with link to full policy) or by providing it in additional languages relevant to Valdorra's client base.

---

## 9. Enforcement and Fines (Articles 83 and 84)

### Fine tiers

| Tier | Maximum Fine | Applies To |
|---|---|---|
| **Lower tier** | EUR 10 million or 2% of global annual turnover (whichever is higher) | Violations of: controller/processor obligations (Arts. 8, 11, 25-39, 42, 43), including data protection by design, DPIA obligations, record-keeping, security measures, breach notification, and DPO requirements. |
| **Upper tier** | EUR 20 million or 4% of global annual turnover (whichever is higher) | Violations of: core principles (Art. 5), lawfulness of processing (Art. 6), consent conditions (Art. 7), special categories (Art. 9), data subject rights (Arts. 12-22), and international transfer rules (Ch. V). |

### Context for Valdorra

Fines are not the only enforcement risk:

- **Reputational damage:** A published IMY decision naming Valdorra would damage client trust. As an IT consulting firm, clients expect us to handle data properly — a GDPR finding would directly undermine our professional credibility.
- **Client contractual consequences:** Many client contracts include GDPR compliance warranties. A breach could trigger contract penalties or termination.
- **Corrective orders:** IMY can order Valdorra to stop processing, which could halt business operations.
- **Individual compensation claims:** Data subjects can claim compensation for material and non-material damage (Art. 82).

### Assessment context

When evaluating a feature, note which fine tier would apply if the identified issues constitute a GDPR violation. Blocking issues in Sections 1-6 and 8 (lawful basis, consent, data subject rights, minimisation, transfers, automated decisions, transparency) generally fall under the upper tier. Blocking issues in Section 7 (DPIA) fall under the lower tier. This distinction provides context for the severity of identified risks but does not change the assessment methodology — a blocking issue is grounds for rejection regardless of which fine tier applies.

---

## 10. Valdorra Feature Assessment Checklist

Use this checklist when reviewing any proposed feature, tool, process change, or new data processing activity. For each item, determine whether the requirement is met, partially met, or not met, and classify any issues by tier (blocking, significant, advisory).

If a requirement cannot be assessed because information is missing from the proposal, treat it according to the rules in "Handling Incomplete Information" at the top of this document.

### A. Data Inventory

- IF the feature processes personal data: identify which categories of data, which categories of data subjects, the source (directly from subject / from client / from third party / system-generated), the storage location (system and geographic region), and who has access (roles, teams, third parties).
- IF any of the above cannot be determined from the proposal: **BLOCKING.** The proposal does not contain sufficient information to assess compliance. Grounds for rejection until the information is provided.

### B. Lawful Basis

- IF no legal basis is stated or identifiable: **BLOCKING.** Grounds for rejection.
- IF the stated legal basis is consent: verify all six conditions (free, specific, informed, unambiguous, documented, withdrawable). Failure on any single condition: **BLOCKING.**
- IF the stated legal basis is legitimate interest: verify that a Legitimate Interest Assessment exists or is referenced. If absent: **SIGNIFICANT.** Remediation required before deployment.
- IF the stated legal basis does not match the described processing (e.g., "contract performance" for activities beyond contract necessity): **BLOCKING.** Grounds for rejection.
- IF the processing involves special category data (Art. 9): verify both an Art. 9(2) exception and an Art. 6(1) basis. If either is missing: **BLOCKING.**

### C. Purpose and Minimisation

- IF the feature repurposes existing data for a new, incompatible purpose without a new legal basis: **BLOCKING.** Grounds for rejection.
- IF the feature collects data fields with no documented purpose: **SIGNIFICANT.** Remove unnecessary fields or provide justification.
- IF the stated purpose could be achieved with less data, anonymised data, or aggregated data: **SIGNIFICANT.** Justify why the less invasive approach is insufficient, or adopt it.
- IF the feature defaults to maximum data collection rather than minimum: **SIGNIFICANT.** Invert the default.
- IF the feature collects only necessary data but pseudonymisation or aggregation could further reduce risk: **ADVISORY.**

### D. Data Subject Rights

- IF the feature stores personal data in a system or format from which it cannot be retrieved or deleted: **BLOCKING.** Grounds for rejection.
- IF deletion is soft-delete only with no documented hard-deletion process: **BLOCKING.** Grounds for rejection.
- IF the feature introduces a new data store not covered by Valdorra's subject access request process: **SIGNIFICANT.** Add it to the SAR process before deployment.
- IF data portability applies (consent or contract basis, data provided by subject) and the feature does not support structured export: **SIGNIFICANT.** Remediation required.
- IF the feature makes access requests more time-consuming but not impossible: **ADVISORY.**

### E. Transparency

- IF the feature collects data directly from individuals and no privacy notice is provided: **BLOCKING.** Grounds for rejection.
- IF the feature obtains data from third-party sources and no Art. 14 notification process exists: **BLOCKING.** Grounds for rejection.
- IF a privacy notice exists but does not cover the proposed processing: **SIGNIFICANT.** Update the privacy notice before deployment.
- IF the notice exists and covers the processing but could be clearer: **ADVISORY.**

### F. Third Parties and Transfers

- IF a new third-party processor is involved and no Data Processing Agreement (Art. 28) is in place: **BLOCKING.** Grounds for rejection.
- IF data is transferred outside the EU/EEA with no valid transfer mechanism (adequacy, SCCs, DPF, or valid derogation): **BLOCKING.** Grounds for rejection.
- IF a transfer mechanism exists but no Transfer Impact Assessment has been completed: **SIGNIFICANT.** Remediation required before deployment.
- IF an EU-based alternative exists that would eliminate the transfer entirely: **ADVISORY.**

### G. Security

- IF the proposal does not address encryption, access controls, or breach response: treat as missing information on a core requirement. **BLOCKING** if the feature handles sensitive data or large volumes of personal data. **SIGNIFICANT** otherwise.
- IF access is not restricted by role (principle of least privilege): **SIGNIFICANT.** Remediation required.
- IF no breach response plan covers the system: **SIGNIFICANT.** Remediation required.
- IF security measures are adequate but could be strengthened: **ADVISORY.**

### H. Retention

- IF no retention period is stated or determinable: **BLOCKING.** Grounds for rejection.
- IF a retention period is stated but has no documented basis (legal requirement, contractual need, or justified legitimate interest): **SIGNIFICANT.** Provide justification or adopt a standard retention period.
- IF no automated or scheduled deletion process exists: **SIGNIFICANT.** Remediation required.
- IF retention is defined and justified but could be shortened without operational impact: **ADVISORY.**

### I. DPIA and Risk

- IF the feature meets two or more of the nine EDPB criteria (see Section 7) and no DPIA has been completed: **BLOCKING.** Grounds for rejection.
- IF a DPIA identifies high residual risk and no prior consultation with IMY has occurred: **BLOCKING.** Grounds for rejection.
- IF a DPIA is in progress but not yet complete: **SIGNIFICANT.** Approval contingent on completion before deployment.
- IF the feature meets one criterion: **ADVISORY.** Document the risk factor.

### J. Documentation

- IF the Record of Processing Activities has not been updated to include this processing: **SIGNIFICANT.** Required before deployment.
- IF a new processor is involved and the processor register has not been updated: **SIGNIFICANT.** Required before deployment.
- IF the privacy policy needs updating but the update has not been made: **SIGNIFICANT.** Required before or concurrent with deployment.
- IF documentation is complete: requirement met.

---

## Quick Reference: Valdorra's Current Processing Activities

| Activity | Legal Basis | Key Systems | Retention |
|---|---|---|---|
| Client consulting delivery | Art. 6(1)(b) contract | M365, Azure, Salesforce, Slack, Asana | Engagement + 3 years |
| CRM relationship management | Art. 6(1)(f) legitimate interest | Salesforce | Relationship + 2 years |
| Invoicing and accounting | Art. 6(1)(c) legal obligation | Fortnox | 7 years (Bokforingslagen) |
| Employee HR management | Art. 6(1)(b) contract + 6(1)(c) legal obligation | Personio, Fortnox | Employment + 10 years |
| Recruitment | Art. 6(1)(b) pre-contractual | Personio, M365 | 2 years with consent |
| Internal communications | Art. 6(1)(f) legitimate interest | Slack, M365 | Duration of employment |
| Project management | Art. 6(1)(f) legitimate interest | Asana | Project + 1 year |
| Website analytics | Art. 6(1)(a) consent | Google Analytics | 26 months |
| Newsletter and marketing | Art. 6(1)(a) consent | Mailchimp | Until consent withdrawn |
| IT security logging | Art. 6(1)(f) legitimate interest | Azure, M365 | 12 months |

---

## Worked Example: Assessing a Feature Proposal

**Proposal:** "We want to add a lead scoring feature to Salesforce that automatically ranks prospective clients based on their engagement with our newsletter (Mailchimp open/click data), website behaviour (Google Analytics), and CRM interaction history (Salesforce). The score will determine which prospects our sales team prioritises for outreach."

### Assessment

**A. Data Inventory:**
Data subjects: prospective clients. Data categories: name, email, company, newsletter engagement metrics, website behaviour, CRM interaction history. Source: directly from subjects (newsletter signup, website visits) and system-generated (engagement metrics, scores). Storage: Salesforce (EU-Germany). Access: sales team, marketing team. — *Requirement met.*

**B. Lawful Basis:**
The proposal does not state a legal basis. Newsletter data was collected under consent (Art. 6(1)(a)) for the purpose of sending newsletters. Website analytics data was collected under consent for analytics. CRM data exists under legitimate interest for relationship management. Combining these datasets for lead scoring is a new purpose not covered by the original consent or the existing legitimate interest basis. — **BLOCKING. No valid legal basis identified for the combined processing. The original consent for newsletter and analytics does not extend to lead scoring. A new legal basis is required — likely legitimate interest with a documented LIA, or fresh consent. Grounds for rejection.**

**C. Purpose and Minimisation:**
The feature combines three separate data sources for a purpose (lead scoring) not covered by any of their original collection purposes. This is a purpose limitation issue under Art. 5(1)(b). Additionally, the proposal does not specify which data fields are used in scoring — it may include more data than necessary. — **BLOCKING (purpose limitation violation, see B above). SIGNIFICANT: data fields used in scoring must be specified and justified.**

**D. Data Subject Rights:**
If the score produces "significant effects" (determines who receives sales attention and who does not), Art. 22 may apply. However, the score informs human prioritisation rather than making a solely automated decision — the sales team decides who to contact. This is likely not a solely automated decision, but the boundary is arguable. — **SIGNIFICANT: document that the score is advisory only and that sales staff exercise independent judgment. If the score becomes the sole basis for contact decisions, Art. 22 applies and human review mechanisms would be required.**

**E. Transparency:**
Prospects were not informed at the time of newsletter signup or website visit that their data would be used for lead scoring. The current privacy notice does not cover this processing. — **SIGNIFICANT: the privacy notice must be updated before deployment to include lead scoring as a purpose. For existing newsletter subscribers, retrospective notice is required.**

**F. Third Parties and Transfers:**
Data flows from Mailchimp (US) and Google Analytics (US) into Salesforce (EU). The transfer risk exists for the original data collection (already covered by existing DPF + SCCs for Mailchimp and consent + DPF for Google Analytics). The combined processing occurs in Salesforce EU. No new transfer is introduced. — *Requirement met for transfers.* However: combining Mailchimp and Google Analytics data into Salesforce may require updating the Data Processing Agreements to cover this data flow. — **SIGNIFICANT: verify DPAs cover this data sharing between processors.**

**G. Security:**
Salesforce access controls and encryption are already in place. The lead score itself is not particularly sensitive. — *Requirement met,* assuming existing Salesforce security measures apply.

**H. Retention:**
No retention period is stated for the lead scores. — **BLOCKING: a retention period must be defined. Prospects who are not converted should not have scores retained indefinitely. Grounds for rejection until a retention period is specified.**

**I. DPIA:**
The feature involves: (1) evaluation/scoring, (2) matching/combining datasets, and (7) data concerning individuals who may not have a direct relationship with Valdorra (prospects). Three of the nine EDPB criteria are met. A DPIA is required. — **BLOCKING: no DPIA has been completed. Grounds for rejection.**

**J. Documentation:**
The Record of Processing Activities does not include lead scoring as a processing activity. — **SIGNIFICANT: must be updated before deployment.**

### Summary

This proposal has **four blocking issues** (no legal basis, no retention period, no DPIA, purpose limitation) and **five significant issues** (unspecified data fields, Art. 22 boundary documentation, privacy notice update, DPA verification, ROPA update). **Grounds for rejection.** The proposal requires substantial revision before it can be reconsidered.

---

*Document version 2.0 — Effective 1 March 2026 — Owner: Data Protection Officer — Classification: Internal*
