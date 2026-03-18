"""
Disease-specific clinical prior knowledge for zero-shot EHR prediction.

Each entry provides a concise but information-dense clinical summary meant to
be injected into the LLM prompt so the model can reason about relevant signals
even when those signals appear indirectly in the structured event history.

Sources: standard clinical guidelines (ACR, AGA, USPSTF), epidemiology literature.
"""

TASK_PRIORS: dict[str, str] = {

    # -----------------------------------------------------------------------
    # Celiac disease (K90.0)
    # -----------------------------------------------------------------------
    "new_celiac": """\
Celiac disease is an immune-mediated enteropathy triggered by dietary gluten (wheat, barley, rye).

Key risk factors:
- Family history of celiac disease (first-degree relatives: ~10% risk)
- Other autoimmune conditions: Type 1 diabetes mellitus, autoimmune thyroid disease \
(Hashimoto's, Graves'), Sjögren's syndrome, autoimmune liver disease
- Chromosomal anomalies: Down syndrome (trisomy 21), Turner syndrome, Williams syndrome
- Female sex (slightly higher prevalence)

Clinical signals to look for in the EHR:
- Chronic or recurrent gastrointestinal complaints: diarrhea, steatorrhea, bloating, \
abdominal pain, nausea, weight loss
- Malabsorption consequences: iron-deficiency anemia (refractory to oral iron), \
folate/B12 deficiency, vitamin D/calcium deficiency, osteoporosis or low bone density
- Dermatitis herpetiformis (intensely pruritic vesicular rash — a skin manifestation of celiac)
- Short stature or delayed growth in children/adolescents
- Elevated liver enzymes (cryptogenic hypertransaminasemia) without other explanation
- Peripheral neuropathy or cerebellar ataxia of unclear etiology
- Recurrent aphthous oral ulcers
- Unexplained infertility or recurrent miscarriages

Laboratory and procedure clues:
- Positive anti-tissue transglutaminase IgA (tTG-IgA) or anti-endomysial IgA (EMA)
- Total serum IgA measurement (to rule out IgA deficiency, which causes false-negative serology)
- Upper endoscopy / small bowel biopsy showing villous atrophy (Marsh III)
- Duodenal biopsy reports mentioning intraepithelial lymphocytosis

Key insight: Many patients present with atypical or extra-intestinal features rather than \
classic GI symptoms. Refractory anemia and autoimmune comorbidities are strong indirect signals.""",

    # -----------------------------------------------------------------------
    # Systemic lupus erythematosus (M32.x)
    # -----------------------------------------------------------------------
    "new_lupus": """\
Systemic lupus erythematosus (SLE) is a chronic, relapsing-remitting autoimmune disease \
affecting multiple organ systems.

Key risk factors:
- Female sex and reproductive age (female:male ratio ~9:1; peak onset 15–45 years)
- African American, Hispanic, Asian, or Native American ancestry (higher prevalence and severity)
- Family history of SLE or other autoimmune diseases
- Certain medications can trigger drug-induced lupus (hydralazine, procainamide, isoniazid, \
minocycline, TNF-alpha inhibitors)

ACR/EULAR classification criteria — signals to look for:
Constitutional: unexplained fever, fatigue, weight loss
Skin/mucosa: malar (butterfly) rash, discoid rash, photosensitivity, oral ulcers, \
non-scarring alopecia, acute cutaneous lupus lesions
Musculoskeletal: non-erosive arthritis (multiple joints, symmetric), arthralgia, myositis
Serositis: pleuritis (pleuritic chest pain, pleural effusion), pericarditis
Renal: proteinuria (>500 mg/day), hematuria, red cell casts — lupus nephritis
Neuropsychiatric: seizures, psychosis, mononeuritis multiplex, transverse myelitis, headache
Hematologic: hemolytic anemia, leukopenia (<4000/µL), lymphopenia (<1000/µL), \
thrombocytopenia (<100,000/µL)

Laboratory clues:
- Positive ANA (anti-nuclear antibody) — highly sensitive but not specific
- Anti-dsDNA (double-stranded DNA antibody) — highly specific for SLE
- Anti-Smith (anti-Sm) antibody — highly specific
- Low complement levels: C3, C4 (consumed during active disease)
- Antiphospholipid antibodies: lupus anticoagulant, anti-cardiolipin, anti-β2-glycoprotein I \
(associated with thrombosis and pregnancy loss)
- Direct Coombs test positive

Important comorbidity patterns:
- Antiphospholipid syndrome: recurrent DVT, pulmonary embolism, stroke at young age, \
recurrent pregnancy loss
- Raynaud's phenomenon
- Sjögren's syndrome overlap
- Increased cardiovascular risk (premature atherosclerosis, pericarditis)
- Frequent infections due to immune dysregulation and immunosuppressive therapy

Key insight: Young or middle-aged women with multisystem involvement (joint pain + rash + \
cytopenias + renal abnormalities) and serologic autoimmune markers are the highest-risk \
pattern. Isolated findings raise suspicion; combination of organ systems greatly increases \
pre-test probability.""",

}
