import pdb

"""

"""

STOP_WORDS = ["mean living area", "Renovation", "Noise level", "Outlier detection", "Steuerfuss",
              "Tags gruppieren", "Stacked model", "Without Tags"]

STATS_WORD = ["R²-Score:", "MAPE:", "MdAPE:", "Min", "Max", "Max", "Mean", "Median", "Mean"]
def main():
    with open('train.log', 'r') as f:
        lines = f.readlines()
    
    table_name = ""
    table_per = ""
    table_stat = ""
    table_featue = ""
    f_c = 0
    counter = 0
    idx = 0
    for line in lines:
        if line.startswith("BREAK"):
            table_per += """\end{table*}
                """
            counter = 0
            table_stat += """\end{table*}
                """
            with open("./report/attachments/ml_results_{}.tex".format(idx), 'w') as file:
                    file.write(table_per)

            with open("./report/attachments/ml_results2_{}.tex".format(idx), 'w') as file:
                file.write(table_stat)
            idx += 1
            table_per = ""
            table_stat = ""
            continue
        l = line[32:].strip()
        if l.startswith('-') or l.startswith('='):
            continue

        if l.startswith('Statistics for:'):
            table_name = l.split(":")[1].strip().replace("_", "\_")
            if table_name == "adaboost":
                table_name = "AdaBoost"
            elif table_name == "xgb":
                table_name = "XGBoost"
            elif table_name == "ExtraTree\_train":
                table_name = "Extra Trees"
            if counter == 0:
                table_per += """
\\begin{table*}[ht]
\\begin{minipage}{.3\\textwidth}
\centering
\\ra{1.3}
\\resizebox{\\textwidth}{!}{
\\begin{tabular}{@{}lr@{}}
\\toprule
Abweichung in \% & Abdeckung in \%\\\\
\midrule"""
                table_stat += """
\\begin{table*}[ht]
\\begin{minipage}{.3\\textwidth}
\centering
\\ra{1.3}
\\resizebox{\\textwidth}{!}{
\\begin{tabular}{@{}lr@{}}
\\toprule
Name & Wert in \\\\
\midrule"""

            if counter >= 1:
                table_per += """\\begin{minipage}{.3\\textwidth}
\centering
\\ra{1.3}
\\resizebox{\\textwidth}{!}{
\\begin{tabular}{@{}lr@{}}
\\toprule
Abweichung in \% & Abdeckung in \%\\\\
\midrule"""
                table_stat += """\\begin{minipage}{.3\\textwidth}
\centering
\\ra{1.3}
\\resizebox{\\textwidth}{!}{
\\begin{tabular}{@{}lr@{}}
\\toprule
Name & Wert in \\\\
\midrule"""
        if l.split()[0] in STATS_WORD:
            name, value = l.split(':')
            table_stat += """
            {} & {:.2f}\\\\""".format(name.replace(":", "").replace("%", ""), float(value.split("%")[0]))

        if l.startswith('I'):
            percent, value = l.split(':')

            table_per += """
{} & {:.2f}\\\\""".format(percent.split(" ")[-1], float(value.split("%")[0]))

        if l.startswith('PLOT NR'):
            if counter <= 1:
                table_per += """
\\bottomrule
\end{tabular}}
\caption{""" +table_name+"""}
\end{minipage}
"""
                table_stat += """
\\bottomrule
\end{tabular}}
\caption{""" +table_name+"""}
\end{minipage}
"""
            elif counter >= 2:
                table_per += """
\\bottomrule
\end{tabular}}
\caption{""" +table_name+"""}
\end{minipage}
\end{table*}
                """

                table_stat += """
\\bottomrule
\end{tabular}}
\caption{""" +table_name+"""}
\end{minipage}
\end{table*}
                """
                with open("./report/attachments/ml_results_{}.tex".format(idx), 'w') as file:
                    file.write(table_per)

                with open("./report/attachments/ml_results2_{}.tex".format(idx), 'w') as file:
                    file.write(table_stat)
                
                idx += 1
                table_per = ""
                table_stat = ""
            counter = (counter+1) % 3

if __name__ == "__main__":
    main()