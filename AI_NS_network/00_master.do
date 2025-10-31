******************************************************
* Project: AI-Powered Classification & Network Mapping
* Script:  00_master.do
* Purpose: Orchestrate full Stata pipeline
* Version: 1.1 (2025-10-27)
******************************************************
version 18.0
clear all
set more off
set rmsg on
capture log close _all

* Relative directories
global DIR_DATA     "data"
global DIR_FIGURES  "figures"
global DIR_OUTPUT   "output"
global DIR_LOGS     "logs"
global DIR_SRC      "src/stata"

* Master log
tempfile ts
local ts = subinstr("`c(current_time)'", ":", "", .)
log using "${DIR_LOGS}/00_master_`c(current_date)'_`ts'.log", text replace

* Run steps in order
do "${DIR_SRC}/01_agreement_FoR_classif.do"
do "${DIR_SRC}/02_network_figures.do"
do "${DIR_SRC}/03_descriptive_stats.do"
do "${DIR_SRC}/04_itsa_network_analysis.do"

log close _all
