******************************************************
* Project: AI-Powered Classification & Network Mapping
* Script:  04_itsa_network_analysis_min.do
* Purpose: Produce manuscript ITSA figures only
* Author:  JF
* Version: 1.0 (2025-10-31)
******************************************************
version 18.0
clear all
set more off
capture log close _all

* Standard directories (edit if needed)
global DIR_DATA     "data"
global DIR_FIGURES  "figures"
global DIR_OUTPUT   "output"
global DIR_LOGS     "logs"

log using "${DIR_LOGS}/04_itsa_network_analysis_min.log", text replace

* =========================
* FIGURE A: OVERALL (CLINICAL MILESTONES: 1945, 1981, 1998)
* =========================
import excel "${PROJECT_ROOT}", sheet("Sheet1") firstrow clear
collapse Degree-NumberofUniqueIDs, by(Year)
tsset Year
itsa NumberofUniqueIDs, single trperiod(1945; 1981; 1998) lag(1) ///
    figure posttrend family(poisson) link(log) ci eform
graph export "${DIR_FIGURES}/Fig_ITSA_overall_clinical.png", replace

* =========================
* FIGURE B: OVERALL (TECH MILESTONES: 1971, 1999, 2010)
* =========================
* Reuse same collapsed data
itsa NumberofUniqueIDs, single trperiod(1971; 1999; 2010) lag(1) ///
    figure posttrend family(poisson) link(log) ci eform
graph export "${DIR_FIGURES}/Fig_ITSA_overall_tech.png", replace

* Helper: field labels
program define _label_fields
    label define fields  1 Agr_Vet_Food 2 Arts 3 Bio 4 Biomed_Clinical 5 Chemical ///
                         6 Comput 7 Eco 8 Educ 9 Eng 10 Envir 11 Envir_design      ///
                         12 Health 13 Hiso_Archaeo 14 Human 15 Indig 16 Lang 17 Law ///
                         18 Math 19 Philo 20 Physic 21 Psycho, replace
end

* =========================
* FIGURE C: FIELD-SPECIFIC — HEALTH (Field 12), clinical milestones
* =========================
import excel "${PROJECT_ROOT}", sheet("Sheet1") firstrow clear
rename Unique_IDs_Count Rec_year_field
egen Field_n = group(Field)
_label_fields
label values Field_n fields
tsset Field_n Year
keep if Field_n == 12
itsa Rec_year_field, single treatid(12) trperiod(1945; 1981; 1998) lag(1) ///
    figure posttrend family(poisson) link(log) ci eform
graph export "${DIR_FIGURES}/Fig_ITSA_field12_health_clinical.png", replace

* =========================
* FIGURE D: FIELD-SPECIFIC — BIOLOGICAL (Field 3), clinical milestones
* =========================
import excel "${PROJECT_ROOT}", sheet("Sheet1") firstrow clear
rename Unique_IDs_Count Rec_year_field
egen Field_n = group(Field)
_label_fields
label values Field_n fields
tsset Field_n Year
keep if Field_n == 3
itsa Rec_year_field, single treatid(3) trperiod(1945; 1981; 1998) lag(1) ///
    figure posttrend family(poisson) link(log) ci eform
graph export "${DIR_FIGURES}/Fig_ITSA_field3_bio_clinical.png", replace

* =========================
* FIGURE E: FIELD-SPECIFIC — BIOMED/CLINICAL (Field 4), clinical milestones
* =========================
import excel "${PROJECT_ROOT}", sheet("Sheet1") firstrow clear
rename Unique_IDs_Count Rec_year_field
egen Field_n = group(Field)
_label_fields
label values Field_n fields
tsset Field_n Year
keep if Field_n == 4
itsa Rec_year_field, single treatid(4) trperiod(1945; 1981; 1998) lag(1) ///
    figure posttrend family(poisson) link(log) ci eform
graph export "${DIR_FIGURES}/Fig_ITSA_field4_biomed_clinical.png", replace

* =========================
* FIGURE F: FIELD-SPECIFIC — CHEMICAL (Field 5), clinical milestones
* =========================
import excel "${PROJECT_ROOT}", sheet("Sheet1") firstrow clear
rename Unique_IDs_Count Rec_year_field
egen Field_n = group(Field)
_label_fields
label values Field_n fields
tsset Field_n Year
keep if Field_n == 5
itsa Rec_year_field, single treatid(5) trperiod(1945; 1981; 1998) lag(1) ///
    figure posttrend family(poisson) link(log) ci eform
graph export "${DIR_FIGURES}/Fig_ITSA_field5_chemical_clinical.png", replace

* =========================
* FIGURE G: COMPARATIVE ITSA — Biomed/Clinical (treatid=4) vs Chemical (contid=5)
* =========================
import excel "${PROJECT_ROOT}", sheet("Sheet1") firstrow clear
rename Unique_IDs_Count Rec_year_field
egen Field_n = group(Field)
_label_fields
label values Field_n fields
tsset Field_n Year
keep if inlist(Field_n, 3, 4, 5, 12)

* G1: Clinical milestones
itsa Rec_year_field, treatid(4) contid(5) trperiod(1945; 1981; 1998) lag(1) ///
    figure posttrend family(poisson) link(log) ci eform
graph export "${DIR_FIGURES}/Fig_ITSA_comp_field4_vs5_clinical.png", replace

* G2: Tech milestones
itsa Rec_year_field, treatid(4) contid(5) trperiod(1971; 1999; 2010) lag(1) ///
    figure posttrend family(poisson) link(log) ci eform
graph export "${DIR_FIGURES}/Fig_ITSA_comp_field4_vs5_tech.png", replace

log close _all

