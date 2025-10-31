******************************************************
* Project: AI-Powered Classification & Network Mapping
* Script:  02_network_figures_min.do
* Purpose: Produce ONLY manuscript network figures
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

log using "${DIR_LOGS}/02_network_figures_min.log", text replace

*==================================================
* Load network metrics (EDIT to your actual file)
* Expect columns: Year, NumberofUniqueIDs, NumberofNodes, NumberofEdges,
* AverageDegreeCentrality, AverageKatzCentrality, AverageClusteringCoefficient,
* AverageBetweennessCentrality, NetworkDiameter, NetworkDensity, AveragePathLength,
* Field (for field-specific panels)
*==================================================
* Example if XLSX:
* import excel "${DIR_DATA}/network_metrics.xlsx", sheet("Sheet1") firstrow clear
import excel "${PROJECT_ROOT}", sheet("Sheet1") firstrow clear

*==================================================
* FIGURE N1 — Overall size/connectivity vs records
*==================================================
* (Nodes & edges on axis 1, records on axis 2)
twoway ///
    (line NumberofNodes  Year, lwidth(medthick)) ///
    (line NumberofEdges  Year, lwidth(medthick)) ///
    (line NumberofUniqueIDs Year, yaxis(2) lwidth(medthick)), ///
    ytitle("Nodes / Edges", axis(1)) ///
    ytitle("Number of records", axis(2)) ///
    xtitle("Year") ///
    legend(order(1 "Nodes" 2 "Edges" 3 "Records") pos(6) ring(0)) ///
    xlabel(1926(20)2024)

graph export "${DIR_FIGURES}/Fig_network_overall_nodes_edges_records.png", replace

*==================================================
* FIGURE N2 — 5-year aggregated network metrics
*==================================================
gen period = 5 * floor(Year/5)
preserve
collapse ///
    (mean) NetworkDiameter NetworkDensity AveragePathLength AverageClusteringCoefficient, ///
    by(period)

twoway ///
    (line NetworkDiameter             period, lwidth(medthick)) ///
    (line NetworkDensity              period, lwidth(medthick)) ///
    (line AveragePathLength           period, lwidth(medthick)) ///
    (line AverageClusteringCoefficient period, lwidth(medthick)), ///
    legend(order(1 "Diameter" 2 "Density" 3 "Avg path length" 4 "Avg clustering") pos(6) ring(0)) ///
    xtitle("5-year period") ///
    xlabel(1925(10)2025)

graph export "${DIR_FIGURES}/Fig_network_metrics_5yr.png", replace
restore

*==================================================
* FIELD-SPECIFIC CENTRALITY TRENDS (4 key FoR)
* Expected long-format rows with Field string equal to:
* "Chemical", "Bio", "Biomed_Clinical", "Health"
*==================================================

* Ensure period for field plots
capture confirm variable period
if _rc gen period = 5 * floor(Year/5)

* Keep only the 4 main fields used in the manuscript
keep if inlist(Field, "Chemical", "Bio", "Biomed_Clinical", "Health")

* Aggregate to 5-year periods per field to reduce noise (as in text)
preserve
collapse (mean) DegreeCentrality=AverageDegreeCentrality ///
                 KatzCentrality=AverageKatzCentrality ///
                 ClusteringCoefficient=AverageClusteringCoefficient ///
                 BetweennessCentrality=AverageBetweennessCentrality, ///
         by(period Field)

* ---- FIGURE N3 — Degree Centrality over time by field
twoway ///
    (line DegreeCentrality period if Field=="Chemical",          lwidth(medthick)) ///
    (line DegreeCentrality period if Field=="Bio",               lwidth(medthick)) ///
    (line DegreeCentrality period if Field=="Biomed_Clinical",   lwidth(medthick)) ///
    (line DegreeCentrality period if Field=="Health",            lwidth(medthick)), ///
    title("Degree centrality (5-year means)") ///
    xtitle("5-year period") ytitle("Centrality") ///
    legend(order(1 "Chemical" 2 "Bio" 3 "Biomed/Clinical" 4 "Health") pos(6) ring(0)) ///
    xlabel(1925(10)2025)
graph export "${DIR_FIGURES}/Fig_field_degree_5yr.png", replace

* ---- FIGURE N4 — Katz Centrality over time by field
twoway ///
    (line KatzCentrality period if Field=="Chemical",          lwidth(medthick)) ///
    (line KatzCentrality period if Field=="Bio",               lwidth(medthick)) ///
    (line KatzCentrality period if Field=="Biomed_Clinical",   lwidth(medthick)) ///
    (line KatzCentrality period if Field=="Health",            lwidth(medthick)), ///
    title("Katz centrality (5-year means)") ///
    xtitle("5-year period") ytitle("Centrality") ///
    legend(order(1 "Chemical" 2 "Bio" 3 "Biomed/Clinical" 4 "Health") pos(6) ring(0)) ///
    xlabel(1925(10)2025)
graph export "${DIR_FIGURES}/Fig_field_katz_5yr.png", replace

* ---- FIGURE N5 — Clustering Coefficient over time by field
twoway ///
    (line ClusteringCoefficient period if Field=="Chemical",          lwidth(medthick)) ///
    (line ClusteringCoefficient period if Field=="Bio",               lwidth(medthick)) ///
    (line ClusteringCoefficient period if Field=="Biomed_Clinical",   lwidth(medthick)) ///
    (line ClusteringCoefficient period if Field=="Health",            lwidth(medthick)), ///
    title("Clustering coefficient (5-year means)") ///
    xtitle("5-year period") ytitle("Clustering") ///
    legend(order(1 "Chemical" 2 "Bio" 3 "Biomed/Clinical" 4 "Health") pos(6) ring(0)) ///
    xlabel(1925(10)2025)
graph export "${DIR_FIGURES}/Fig_field_clustering_5yr.png", replace

* ---- FIGURE N6 — Betweenness Centrality over time by field
twoway ///
    (line BetweennessCentrality period if Field=="Chemical",          lwidth(medthick)) ///
    (line BetweennessCentrality period if Field=="Bio",               lwidth(medthick)) ///
    (line BetweennessCentrality period if Field=="Biomed_Clinical",   lwidth(medthick)) ///
    (line BetweennessCentrality period if Field=="Health",            lwidth(medthick)), ///
    title("Betweenness centrality (5-year means)") ///
    xtitle("5-year period") ytitle("Centrality") ///
    legend(order(1 "Chemical" 2 "Bio" 3 "Biomed/Clinical" 4 "Health") pos(10) ring(0)) ///
    xlabel(1925(10)2025)
graph export "${DIR_FIGURES}/Fig_field_betweenness_5yr.png", replace
restore

log close _all
