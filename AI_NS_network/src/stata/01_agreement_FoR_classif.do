******************************************************
* Project: AI-Powered Classification & Network Mapping
* Script:  01_agreement_FoR_classif.do
* Purpose: Reproducible Stata workflow for the paper
* Author:  JF
* Version: 1.0 (2025-10-27)
*****************************************************
version 18.0
clear all
set more off
set rmsg on
capture log close _all
quietly: cd "`c(pwd)'"

* -------- Project root (do not change here) ---------
* If you need an absolute path, you can set a root like:
* global PROJECT_ROOT "/mnt/data/github_release"
* But by default we use relative paths.
* ----------------------------------------------------

* Standard directories (relative to project root)
global DIR_DATA     "data"
global DIR_FIGURES  "figures"
global DIR_OUTPUT   "output"
global DIR_LOGS     "logs"

* Create timestamped log
tempfile ts
local ts = subinstr("`c(current_time)'", ":", "", .)
log using "${DIR_LOGS}/01_agreement_FoR_classif_`c(current_date)'_`ts'.log", text replace


set seed 123456
 u "${PROJECT_ROOT}" , clear
*_________________________________________________________________________________________
*EVALUATE AGREEMENT
*_________________________________________________________________________________________

/////
/* 
DonneesCorrelees_2024.pdf CAS OPTION DONNEES CORRELEES

La répétabilité (repeatability) s'intéresse à quel degré des mesures répétées sous les
mêmes conditions fournissent des résultats similaires:
- La concordance (agreement) évalue à quel point les résultats de mesures répétées sont
proches, en estimant l' erreur de mesure --> évaluer l'accord
- La fiabilité (reliability) évalue à quel point les objets étudiés, souvent des personnes,
peuvent être distinguées les unes des autres en dépit de l'erreur de mesure. Dans ces
situations, l'erreur de mesure est confrontée à la variabilité entre les personnes --> pour discrimier /pas notre cas

Repeatability: 1) Pourcentage de concordance 2) coefficient kappa

https://www.reed.edu/psychology/stata/gs/tutorials/kappa.html
Kappa goes from zero (no agreement) to one (perfect agreement). Stata suggests the following guidelines from Landis & Koch (1977) as to what agreement level a particular kappa value constitutes:

0.0 - .20: slight
.21 - .40: fair
.41 - .60: moderate
.61 - .90: substantial
.81 - 1: almost perfect

*/

// We consider two raters blinded (GPT 2 times but independant of each other)
*Kappa with the Same Two Raters For All Observations
kap Biomed_Clinical Biomed_Clinical_v2 , tab 
/*

           |      3 FoR_yn_v2
  3 FoR_yn |         0          1 |     Total
-----------+----------------------+----------
         0 |        31          3 |        34 
         1 |         5      4,607 |     4,612 
-----------+----------------------+----------
     Total |        36      4,610 |     4,646 
             Expected
Agreement   agreement     Kappa   Std. err.         Z      Prob>Z
-----------------------------------------------------------------
  99.83%      98.50%     0.8848     0.0147      60.34      0.0000

*/

*If each GPT had made his determination randomly (but with probabilities equal to the overall proportions), we would expect the two raters to agree on 98.50% of the evaluations. In fact, they agreed on 99.83% of the cases, or 88.48% of the way between random agreement and perfect agreement. The amount of agreement indicates that we can reject the hypothesis that they are making their determinations randomly.


kap Bio Bio_v2 , tab 

kap Health Health_v2 , tab

label variable Agr_Vet_Food_v2 "Agri, Vet & Food Sciences"
label variable Bio_v2 "Biological Sciences"
label variable Biomed_Clinical_v2 "Biomedical & Clinical Sciences"
label variable Envir_design_v2 "Built Env. & Design"
label variable Chemical_v2 "Chemical Sciences"
label variable Com_v2 "Commerce & Tourism"
label variable Arts_v2 "Creative Arts & Writing"
label variable Earth_v2 "Earth Sciences"
label variable Eco_v2 "Economics"
label variable Educ_v2 "Education"
label variable Eng_v2 "Engineering"
label variable Envir_v2 "Environmental Sciences"
label variable Health_v2 "Health Sciences"
label variable Hiso_Archaeo_v2 "History & Archaeology"
label variable Human_v2 "Human Society"
label variable Indig_v2 "Indigenous Studies"
label variable Comput_v2 "Computing Sciences"
label variable Lang_v2 "Language & Culture"
label variable Law_v2 "Law & Legal Studies"
label variable Math_v2 "Mathematical Sciences"
label variable Philo_v2 "Philosophy & Religion"
label variable Physic_v2 "Physical Sciences"
label variable Psycho_v2 "Psychology"

label variable Agr_Vet_Food "Agri, Vet & Food Sciences"
label variable Bio "Biological Sciences"
label variable Biomed_Clinical "Biomedical & Clinical Sciences"
label variable Envir_design "Built Env. & Design"
label variable Chemical "Chemical Sciences"
label variable Com "Commerce & Tourism"
label variable Arts "Creative Arts & Writing"
label variable Earth "Earth Sciences"
label variable Eco "Economics"
label variable Educ "Education"
label variable Eng "Engineering"
label variable Envir "Environmental Sciences"
label variable Health "Health Sciences"
label variable Hiso_Archaeo "History & Archaeology"
label variable Human "Human Society"
label variable Indig "Indigenous Studies"
label variable Comput "Computing Sciences"
label variable Lang "Language & Culture"
label variable Law "Law & Legal Studies"
label variable Math "Mathematical Sciences"
label variable Philo "Philosophy & Religion"
label variable Physic "Physical Sciences"
label variable Psycho "Psychology"

foreach var of varlist Agr_Vet_Food-Biomed_Clinical Chemical Arts Eco-Psycho {
di `"`: var label `var''"' 
	kap `var' `var'_v2 , tab 
}

kap Com Com_v2 , tab

*CI for proportion of agreeement: "Out of 5411 observations, 5393 were successes (i.e., matching classifications). Give me the proportion and its confidence interval.

*AGRICULTURAL, VETERINARY AND FOOD SCIENCES
4616+23
cii prop 4646 4639

*BIOLOGICAL SCIENCES
di 719+3714
cii prop 4646 4433
*BIOMEDICAL AND CLINICAL SCIENCES
di 31+4607
cii prop 4646 4638
*CHEMICAL SCIENCES
di 4093+470
cii prop 4646 4563
*CREATIVE ARTS AND WRITING
di 4615+28
cii prop 4646 4643
*COMMERCE 
di 4645+1
cii prop 4646 4646
*ECONOMICS
di 4635+10
cii prop 4646 4645
*EDUCATION
di 4633+10
cii prop 4646 4643
*ENGINEERING
di 4637+8
cii prop 4646 4645
*ENVIRONMENTAL SCIENCES
di 4642+4
cii prop 4646 4646
*HEALTH SCIENCES
di 30+4606
cii prop 4646 4636
*HISTORY, HERITAGE AND ARCHAEOLOGY
di 4559+79
cii prop 4646 4638
*HUMAN SOCIETY
di 4125+449
cii prop 4646 4574
*INDIGENOUS STUDIES
di 4637+9
cii prop 4646 4646
*INFORMATION AND COMPUTING SCIENCES
di 4637+9
cii prop 4646 4646
*LANGUAGE, COMMUNICATION AND CULTURE
di 4628+14
cii prop 4646 4642
*LAW AND LEGAL STUDIES
di 4645+1
cii prop 4646 4646
*MATHEMATICAL SCIENCES
di 4614+29
cii prop 4646 4643
*PHILOSOPHY AND RELIGIOUS STUDIES
di 4637+8
cii prop 4646 4645
*PHYSICAL SCIENCES
di 4615+22
cii prop 4646 4637
*PSYCHOLOGY
di 4237+374
cii prop 4646 4611




* Define the fields 
local fields Agr_Vet_Food Bio Biomed_Clinical Chemical Arts Eco Educ Eng  Health Hiso_Archaeo Human Lang Math Philo Physic Psycho

* Start fresh export file
local outfile "reconcile_ids.xlsx"
cap erase "`outfile'"

* Loop through fields
foreach field in `fields' {
    preserve
    * Compare field with its _v2 version
    keep if `field' != `field'_v2
    keep id

    * Export to a new sheet
    export excel using "`outfile'", sheet("`field'") sheetreplace firstrow(variables)

    restore
}



* RECONCILE
* Define the fields 
local fields Agr_Vet_Food Bio Biomed_Clinical Chemical Arts Eco Educ Eng Health Hiso_Archaeo Human Lang Math Philo Physic Psycho
foreach field of local fields {
    import excel "${PROJECT_ROOT}", ///
        sheet("`field'") firstrow clear
    keep id reconciliationML
	drop in 1
    destring id reconciliationML , replace
	rename reconciliationML `field'_rec
	drop if id==.
	cd"${PROJECT_ROOT}"
	save "`field'_rec.dta", replace
}


local fields Agr_Vet_Food Bio Biomed_Clinical Chemical Arts Eco Educ Eng Health Hiso_Archaeo Human Lang Math Philo Physic Psycho
u "${PROJECT_ROOT}" , clear
foreach field of local fields {
cd "${PROJECT_ROOT}"
merge 1:1 id using "`field'_rec.dta"
replace `field'=`field'_rec if  `field'_rec!=.
drop `field'_rec _merge
}
drop *_v2

label variable Agr_Vet_Food "AGRICULTURAL, VETERINARY AND FOOD SCIENCES"
label variable Bio "BIOLOGICAL SCIENCES"
label variable Biomed_Clinical "BIOMEDICAL AND CLINICAL SCIENCES"
label variable Envir_design "BUILT ENVIRONMENT AND DESIGN"
label variable Chemical "CHEMICAL SCIENCES"
label variable Com "COMMERCE, MANAGEMENT, TOURISM AND SERVICES"
label variable Arts "CREATIVE ARTS AND WRITING"
label variable Earth "EARTH SCIENCES"
label variable Eco "ECONOMICS"
label variable Educ "EDUCATION"
label variable Eng "ENGINEERING"
label variable Envir "ENVIRONMENTAL SCIENCES"
label variable Health "HEALTH SCIENCES"
label variable Hiso_Archaeo "HISTORY, HERITAGE AND ARCHAEOLOGY"
label variable Human "HUMAN SOCIETY"
label variable Indig "INDIGENOUS STUDIES"
label variable Comput "INFORMATION AND COMPUTING SCIENCES"
label variable Lang "LANGUAGE, COMMUNICATION AND CULTURE"
label variable Law "LAW AND LEGAL STUDIES"
label variable Math "MATHEMATICAL SCIENCES"
label variable Philo "PHILOSOPHY AND RELIGIOUS STUDIES"
label variable Physic "PHYSICAL SCIENCES"
label variable Psycho "PSYCHOLOGY"

*No FoR
drop if id==9603 | id==8829 | id==8074 | id==9375 | id==8908

*save "${PROJECT_ROOT}" , replace



log close _all
