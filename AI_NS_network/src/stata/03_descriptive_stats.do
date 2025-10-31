******************************************************
* Project: AI-Powered Classification & Network Mapping
* Script:  03_descriptive_stats_min.do
* Purpose: Produce ONLY manuscript descriptive figures/tables
* Author:  JF
* Version: 1.0 (2025-10-31)
******************************************************
version 18.0
clear all
set more off
capture log close _all

* ---- Standard directories (edit if needed) ----
global DIR_DATA     "data"
global DIR_FIGURES  "figures"
global DIR_OUTPUT   "output"
global DIR_LOGS     "logs"

log using "${DIR_LOGS}/03_descriptive_stats_min.log", text replace

*==================================================
* Load master dataset (EDIT to your actual .dta path)
*==================================================
* Example: use "${DIR_DATA}/ns_master.dta", clear
use "${PROJECT_ROOT}", clear

*==================================================
* HARMONIZE PUBLICATION TYPE (merged_type_clean)
*==================================================
replace merged_type_clean = "Article" ///
    if missing(merged_type_clean) & inlist(lens_PublicationType, ///
    "journal article","journal","journal issue","preprint")

replace merged_type_clean = "Chapter" ///
    if missing(merged_type_clean) & inlist(lens_PublicationType,"book","book chapter")

replace merged_type_clean = "Conference" ///
    if missing(merged_type_clean) & lens_PublicationType=="conference proceedings article"

replace merged_type_clean = "Editorial/Comment/Letter" ///
    if missing(merged_type_clean) & inlist(lens_PublicationType,"editorial","letter")

replace merged_type_clean = "Review" ///
    if missing(merged_type_clean) & lens_PublicationType=="review"

replace merged_type_clean = "Other" ///
    if missing(merged_type_clean) & inlist(lens_PublicationType, ///
    "dataset","dissertation","reference entry","report","other")

replace merged_type_clean = "Unknown" if missing(merged_type_clean)

*==================================================
* HARMONIZE LANGUAGE (merged_lang) — minimal rules
*==================================================
replace merged_lang="eng" if emb_ArticleLanguage=="English" & missing(merged_lang)
replace merged_lang="spa" if emb_ArticleLanguage=="Spanish" & missing(merged_lang)
replace merged_lang="por" if emb_ArticleLanguage=="Portuguese" & missing(merged_lang)
replace merged_lang="fre" if emb_ArticleLanguage=="French" & missing(merged_lang)
replace merged_lang="ger" if emb_ArticleLanguage=="German" & missing(merged_lang)
replace merged_lang="chi" if emb_ArticleLanguage=="Chinese" & missing(merged_lang)

* Common mixed cases
replace merged_lang="eng" if inlist(emb_ArticleLanguage, ///
 "English, French","English, Italian","English, Polish","English, Portuguese", ///
 "English, Serbian","English, Spanish","English, Turkish") & missing(merged_lang)
replace merged_lang="ger" if emb_ArticleLanguage=="German, English" & missing(merged_lang)
replace merged_lang="spa" if emb_ArticleLanguage=="Spanish, English" & missing(merged_lang)
replace merged_lang="tur" if emb_ArticleLanguage=="Turkish, English" & missing(merged_lang)

* Fallback to PubMed language codes
replace merged_lang="eng" if pmd_language=="eng" & missing(merged_lang)
replace merged_lang="por" if pmd_language=="por" & missing(merged_lang)
replace merged_lang="spa" if pmd_language=="spa" & missing(merged_lang)

* Last fallback to auto-detect
replace merged_lang="eng" if merged_lang_detect=="en" & missing(merged_lang)
replace merged_lang="fre" if merged_lang_detect=="fr" & missing(merged_lang)
replace merged_lang="ita" if merged_lang_detect=="it" & missing(merged_lang)
replace merged_lang="jpn" if merged_lang_detect=="ja" & missing(merged_lang)
replace merged_lang="rus" if merged_lang_detect=="ru" & missing(merged_lang)
replace merged_lang="chi" if merged_lang_detect=="zh-cn" & missing(merged_lang)
replace merged_lang="und" if missing(merged_lang)

*==================================================
* FILTER to manuscript sample if needed
*==================================================
* If you subset the manuscript sample with sample_in==1, keep it:
* (Uncomment if applicable to your figures/tables)
* keep if sample_in==1

*==================================================
* FIGURES (exported PNGs)
*==================================================
* Fig: Types of records
graph pie, over(merged_type_clean) ///
    title("Types of records") ///
    plabel(_all percent, color(black) size(medium) format(%9.1f))
graph export "${DIR_FIGURES}/Fig_types_of_records.png", replace

* Fig: Records languages
* (If you only want languages for the manuscript subsample, add: if sample_in==1)
catplot merged_lang, ///
    title("Records languages") ///
    blabel(bar, color(black) size(small) format(%9.1f)) percent
graph export "${DIR_FIGURES}/Fig_records_languages.png", replace

*==================================================
* TABLES — DESCRIPTIVES by selected years + loop
*==================================================
* Prepare fields needed by dtable
encode merged_type_clean, gen(merged_type_clean_cat)
encode merged_lang,       gen(merged_lang_cat)

* Count authors per record (detect ; vs , separator)
gen sep_used = cond(strpos(merged_authors,";")>0,";"," ,")
gen number_of_authors = .
replace number_of_authors = length(merged_authors) - ///
    length(subinstr(merged_authors,";","",.)) + 1 if sep_used==";"
replace number_of_authors = length(merged_authors) - ///
    length(subinstr(merged_authors,",","",.)) + 1 if sep_used==" ,"
replace number_of_authors = . if trim(merged_authors)==""
rename number_of_authors merged_authors_nb
drop sep_used

* Records per year
egen tag_nbrec = tag(merged_year id)
egen nb_records_year = total(tag_nbrec), by(merged_year)

* --- TABLES for specific years used in manuscript ---
local targetyears 1916 1926 1946 1966 1986 2006 2024
foreach y of local targetyears {
    dtable i.merged_type_clean_cat i.merged_lang_cat merged_authors_nb nb_records_year ///
        if merged_year==`y', ///
        title("Descriptive statistics for `y'") ///
        sformat("(N=%s)" frequency) ///
        export("${DIR_OUTPUT}/Table_descriptives_`y'.xlsx", replace)
}

* --- OPTIONAL: single combined table for all years in target set ---
preserve
keep if inlist(merged_year, `targetyears')
dtable i.merged_type_clean_cat i.merged_lang_cat merged_authors_nb nb_records_year, ///
    title("Descriptive statistics (selected years)") ///
    sformat("(N=%s)" frequency) ///
    export("${DIR_OUTPUT}/Table_descriptives_selected_years.xlsx", replace)
restore

log close _all

