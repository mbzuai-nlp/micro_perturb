import base64
import textwrap

#---- ICL Examples ----
EXAMPLES = {
    "methodology_result_conflict": [
        {
            "table": textwrap.dedent(
                r"""
                \begin{table*}
                \centering
                \resizebox{\textwidth}{!}{
                \begin{tabular}{ccccccccccccccccccc}
                \toprule
                \multirow{2}{*}{\textbf{Methods}} & \multicolumn{3}{c}{\textbf{Classification Baseline}} & \multicolumn{9}{c}{\cellcolor{gray!20}\textbf{Generative Baselines} (\emph{Single Model Reasoning})} &  \multicolumn{6}{c}{\cellcolor{gray!20}\textbf{Dual-Model Reasoning with Critic Models}}\\ 
                \cmidrule(r){2-4} \cmidrule(l){5-13} \cmidrule(l){14-19} 
                & \multicolumn{3}{c}{PLM Classifier} & \multicolumn{3}{c}{SFT} & \multicolumn{3}{c}{DPO} & \multicolumn{3}{c}{(\texttt{DARS}) Reasoner only} & \multicolumn{3}{c}{GPT-4 as Critic} &  \multicolumn{3}{c}{\textbf{(\texttt{DARS}) Reasoner+Critic}}\\ 
                \cmidrule(l){1-1} \cmidrule(l){2-4} \cmidrule(l){5-13} \cmidrule(l){14-19} 
                \textbf{Datasets} & ACC & F1 & \textbf{QWK} & ACC & F1 & \textbf{QWK} & ACC & F1 & \textbf{QWK} & ACC & F1 & \textbf{QWK}  & ACC & F1 & \textbf{QWK} & ACC$^{\dagger,*}$ & F1$^{\dagger,*}$ & \textbf{QWK}$^{*}$\\
                \cmidrule(r){1-1} \cmidrule(r){2-4} \cmidrule(l){5-7} \cmidrule(l){8-10} \cmidrule(l){11-13} \cmidrule(l){14-16} \cmidrule(l){17-19} 
                \textbf{ASAP 1}    & 0.7767 & 0.7805 & 0.8528 & 0.6968 & 0.7073 & \textbf{0.8277} & 0.6895 & 0.5655 & 0.8051 & 0.6480 & 0.6606 & 0.8073 & 0.5181 & 0.5106 & 0.6349 & \textbf{0.7274} & \textbf{0.7315} & 0.8100 \\
                \textbf{ASAP 2}    & 0.6798 & 0.6817 & 0.8187 & \textbf{0.7324} & \textbf{0.7468} & \textbf{0.8420} & 0.6761 & 0.6783 & 0.8033 & 0.6925 & 0.7074 & 0.8136 & 0.5869 & 0.5636 & 0.6532 & 0.7136 & 0.7303 & 0.8277 \\
                \textbf{ASAP 5}    & 0.8625 & 0.6055 & 0.8187 & 0.8495 & 0.5600 & 0.8203 & 0.8612 & 0.6449 & 0.8001 & 0.8545 & 0.5424 & 0.7766 & 0.8177 & 0.5119 & 0.6340 & \textbf{0.8645} & \textbf{0.6303} & \textbf{0.8326} \\
                \textbf{ASAP 6}    & 0.8891 & 0.6118 & 0.8426 & 0.8314 & 0.5513 & 0.7273 & 0.8314 & 0.5420 & 0.7522 & 0.8280 & 0.5628 & 0.7232 & 0.8130 & 0.4265 & 0.4754 & \textbf{0.8648} & \textbf{0.5988} & \textbf{0.8016} \\
                \textbf{Pty 1} & 0.6787 & 0.6784 & 0.8853 & 0.5236 & 0.5197 & 0.8082 & 0.5236 & 0.4670 & 0.8196 & 0.5551 & 0.5584 & 0.8221 & 0.4134 & 0.3407 & 0.6018 & \textbf{0.5709} & \textbf{0.5653} & \textbf{0.8253} \\
                \textbf{Pty 2} & 0.6224 & 0.6355 & 0.8385 & 0.5459 & 0.5377 & 0.7004 & 0.5561 & 0.5600 & 0.7599 & 0.5765 & 0.5752 & 0.7604 & 0.5357 & 0.5219 & 0.7688 & \textbf{0.6071} & \textbf{0.6059} & \textbf{0.7705} \\
                \cmidrule(r){1-1} \cmidrule(r){2-4} \cmidrule(l){5-7} \cmidrule(l){8-10} \cmidrule(l){11-13} \cmidrule(l){14-16} \cmidrule(l){17-19} 
                \textbf{Overall}  & 0.7515 & 0.6656 & 0.8428 & 0.6966 & 0.6038 & 0.7877 & 0.6897 & 0.5763 & 0.7900 & 0.6925 & 0.6011 & 0.7839 & 0.6141 & 0.4792& 0.6280 & \textbf{0.7247} & \textbf{0.6437} & \textbf{0.8113} \\ 
                \bottomrule
                \end{tabular}
                }
                \caption{\textbf{Comparison of assessment performance across baseline and Reasoner only preference optimization methods.} Generative methods are indicated with a \colorbox{gray!20}{gray background}. All methods were reproduced or trained using the same LLaMA 3B model as the base. We highlighted the highest values for ACC ($\uparrow$), F1 Score ($\uparrow$), and QWK ($\uparrow$) among generative methods in \textbf{bold}. The overall performance is calculated as the average across all datasets. Symbols $\dagger$ and $*$ indicate statistical significance compared to SFT and DPO by each metric, respectively.}
                \label{tab:main_assessment_comparison}
                \end{table*}"""[1:]
            ),
            "sentences": [r"The overall performance is calculated as the average across all datasets."],
            "edits": [{
                "text_span_to_replace": r"\textbf{Overall} & 0.7515 & 0.6656 & 0.8428 & 0.6966 & 0.6038 & 0.7877 & 0.6897 & 0.5763 & 0.7900 & 0.6925 & 0.6011 & 0.7839 & 0.6141 & 0.4792& 0.6280 & \textbf{0.7247} & \textbf{0.6437} & \textbf{0.8113} \\",
                "new_text_span": r"\textbf{Overall} & 0.7515 & 0.6656 & 0.8428 & 0.6966 & 0.6038 & 0.7877 & 0.6897 & 0.5763 & 0.7900 & 0.6925 & 0.6011 & 0.7839 & 0.6141 & 0.4792& 0.6280 & \textbf{0.7247} & \textbf{0.6237} & \textbf{0.8113} \\",
            }],
            "refuse": False,
            "refusal_reasoning": "",
        },
        {
            "table": "",
            "sentences": [],
            "refuse": True,
            "refusal_reasoning": "It is unclear how the values used for the tables were determined, so it is not possible to introduce this type of error into the manuscript."
        }
    ],
    "logically_invalid_interpretation": [
        {
            "table": textwrap.dedent(
                r"""
                \begin{table}[t]
                \centering
                \resizebox{0.48\textwidth}{!}{
                        \begin{tabular}{cc|ccc|c}
                        \toprule
                        Model & Scale & EC & WC  & TC  &\textbf{Avg}\\
                \hline
                            \multicolumn{6}{c}{\textbf{Legal LLM}}\\
                            \hline
                DISC-LawLLM &13B & 12.44 & 2.58 & 18.18 & 11.07 \\
                fuzi.mingcha &6B & 14.00 & 4.12 & 10.10 & 9.41 \\
                LexiLaw &6B  & 4.67 & 2.06 & 5.05 & 3.93\\
                Tailing &7B & 18.89 & 10.31 & 15.15 & 14.78\\
                zhihai &7B & 6.00 & 3.09 & 5.05 & 4.71 \\
                LawGPT\_zh &6B & 5.11 & 3.61 & 5.05 & 4.59\\
                HanFei &7B & 7.56 & 1.55 & 8.08 & 5.73 \\
                \hline
                \hline
                \multicolumn{6}{c}{\textbf{Small-Scale LLM (<10B)}}\\
                \hline
                Deepseek-R1-1.5B &1.5B & 16.00 & 22.16 & 18.18 & 18.78 \\
                Deepseek-R1-7B &7B & 32.22 & 36.08 & 26.26 & 31.52 \\
                GRPO-Base&1.5B & 52.00 & 61.86 & 38.38 & 50.75 \\
                \cdashline{1-6}
                GRPO-Law&1.5B & 59.11 & 66.49 & 46.46 & 57.36 \\
                LexPam&1.5B & \textbf{64.89} & \textbf{69.59} & \textbf{47.47} & \textbf{60.65} \\
                \hline
                \hline
                \multicolumn{6}{c}{\textbf{Larger-Scale LLM ($\geq$10B)}}\\
                \hline
                Deepseek-R1-32B &32B & 70.22 & 68.56 & 48.48 & 62.42 \\
                GPT-4o-mini &- & 60.00 & 59.28 & 51.52 & 56.93 \\
                QwQ-32B-Preview &32B & 73.33 & 79.90 & 59.60 & 70.94 \\
                Deepseek-R1 &671B & 74.67 & 81.96 & 63.64 & 73.42 \\
                        \bottomrule
                        \end{tabular}
                }
                        \caption{Performance Comparison. LexPam (1.5B) aoutperforms all legal LLMs and small-scale reasoning models, and rivals larger-scale models. Bold indicates the best score within the <10B and the legal LLM groups.}
                \label{tab:main results}
                \end{table}"""[1:],
            ),
            "sentences": [r"LexPam achieves an average accuracy of 60.65\%, substantially outperforming all legal-domain models as well as all reasoning models $\leq$10B, such as DeepSeek-R1-1.5B (18.78\%)."],
            "edits": [
                {
                    "text_span_to_replace": r"LexPam&1.5B & \textbf{64.89} & \textbf{69.59} & \textbf{47.47} & \textbf{60.65} \\",
                    "new_text_span": r"LexPam&1.5B & \textbf{64.89} & \textbf{69.59} & \textbf{47.47} & \textbf{57.36} \\",
                }
            ],
            "refuse": False,
            "refusal_reasoning": "",
        },
        {
            "table": "",
            "sentences": [],
            "refuse": True,
            "refusal_reasoning": "The manuscript text does not make any claims that can be formally verified and have their truthfulness contingent on the values in a table, thus it is not possible to introduce this type of error."
        }
    ],
    "semantic_misrepresentation": [
        # {
        #     "table": textwrap.dedent(
        #         r"""
        #         \begin{table}[t]
        #         \centering
        #         \resizebox{0.48\textwidth}{!}{
        #                 \begin{tabular}{cc|ccc|c}
        #                 \toprule
        #                 Model & Scale & EC & WC  & TC  &\textbf{Avg}\\
        #         \hline
        #                     \multicolumn{6}{c}{\textbf{Legal LLM}}\\
        #                     \hline
        #         DISC-LawLLM &13B & 12.44 & 2.58 & 18.18 & 11.07 \\
        #         fuzi.mingcha &6B & 14.00 & 4.12 & 10.10 & 9.41 \\
        #         LexiLaw &6B  & 4.67 & 2.06 & 5.05 & 3.93\\
        #         Tailing &7B & 18.89 & 10.31 & 15.15 & 14.78\\
        #         zhihai &7B & 6.00 & 3.09 & 5.05 & 4.71 \\
        #         LawGPT\_zh &6B & 5.11 & 3.61 & 5.05 & 4.59\\
        #         HanFei &7B & 7.56 & 1.55 & 8.08 & 5.73 \\
        #         \hline
        #         \hline
        #         \multicolumn{6}{c}{\textbf{Small-Scale LLM (<10B)}}\\
        #         \hline
        #         Deepseek-R1-1.5B &1.5B & 16.00 & 22.16 & 18.18 & 18.78 \\
        #         Deepseek-R1-7B &7B & 32.22 & 36.08 & 26.26 & 31.52 \\
        #         GRPO-Base&1.5B & 52.00 & 61.86 & 38.38 & 50.75 \\
        #         \cdashline{1-6}
        #         GRPO-Law&1.5B & 59.11 & 66.49 & 46.46 & 57.36 \\
        #         LexPam&1.5B & \textbf{64.89} & \textbf{69.59} & \textbf{47.47} & \textbf{60.65} \\
        #         \hline
        #         \hline
        #         \multicolumn{6}{c}{\textbf{Larger-Scale LLM ($\geq$10B)}}\\
        #         \hline
        #         Deepseek-R1-32B &32B & 70.22 & 68.56 & 48.48 & 62.42 \\
        #         GPT-4o-mini &- & 60.00 & 59.28 & 51.52 & 56.93 \\
        #         QwQ-32B-Preview &32B & 73.33 & 79.90 & 59.60 & 70.94 \\
        #         Deepseek-R1 &671B & 74.67 & 81.96 & 63.64 & 73.42 \\
        #                 \bottomrule
        #                 \end{tabular}
        #         }
        #                 \caption{Performance Comparison. LexPam (1.5B) aoutperforms all legal LLMs and small-scale reasoning models, and rivals larger-scale models. Bold indicates the best score within the <10B and the legal LLM groups.}
        #         \label{tab:main results}
        #         \end{table}
        #         """[1:],
        #     ),
        #     "sentences": [r"Despite being 1.5B in size, LexPam surpasses GPT-4o-mini (56.93\%) and comes close to 32B-scale models like DeepSeek-R1-32B (62.42\%) and QwQ-32B (70.94\%)."],
        #     "edits": [
        #         {
        #             "text_span_to_replace": r"LexPam&1.5B & \textbf{64.89} & \textbf{69.59} & \textbf{47.47} & \textbf{60.65} \\",
        #             "new_text_span": r"LexPam&1.5B & \textbf{61.00} & 58.30 & \textbf{51.51} & 56.94 \\",
        #         }
        #     ],
        #     "refuse": False,
        #     "refusal_reasoning": ""
        # },
                {
            "table": textwrap.dedent(
                r"""
                \begin{table}[t]
                \centering
                \resizebox{0.48\textwidth}{!}{
                        \begin{tabular}{cc|ccc|c}
                        \toprule
                        Model & Scale & EC & WC  & TC  &\textbf{Avg}\\
                \hline
                            \multicolumn{6}{c}{\textbf{Legal LLM}}\\
                            \hline
                DISC-LawLLM &13B & 12.44 & 2.58 & 18.18 & 11.07 \\
                fuzi.mingcha &6B & 14.00 & 4.12 & 10.10 & 9.41 \\
                LexiLaw &6B  & 4.67 & 2.06 & 5.05 & 3.93\\
                Tailing &7B & 18.89 & 10.31 & 15.15 & 14.78\\
                zhihai &7B & 6.00 & 3.09 & 5.05 & 4.71 \\
                LawGPT\_zh &6B & 5.11 & 3.61 & 5.05 & 4.59\\
                HanFei &7B & 7.56 & 1.55 & 8.08 & 5.73 \\
                \hline
                \hline
                \multicolumn{6}{c}{\textbf{Small-Scale LLM (<10B)}}\\
                \hline
                Deepseek-R1-1.5B &1.5B & 16.00 & 22.16 & 18.18 & 18.78 \\
                Deepseek-R1-7B &7B & 32.22 & 36.08 & 26.26 & 31.52 \\
                GRPO-Base&1.5B & 52.00 & 61.86 & 38.38 & 50.75 \\
                \cdashline{1-6}
                GRPO-Law&1.5B & 59.11 & 66.49 & 46.46 & 57.36 \\
                LexPam&1.5B & \textbf{64.89} & \textbf{69.59} & \textbf{47.47} & \textbf{60.65} \\
                \hline
                \hline
                \multicolumn{6}{c}{\textbf{Larger-Scale LLM ($\geq$10B)}}\\
                \hline
                Deepseek-R1-32B &32B & 70.22 & 68.56 & 48.48 & 62.42 \\
                GPT-4o-mini &- & 60.00 & 59.28 & 51.52 & 56.93 \\
                QwQ-32B-Preview &32B & 73.33 & 79.90 & 59.60 & 70.94 \\
                Deepseek-R1 &671B & 74.67 & 81.96 & 63.64 & 73.42 \\
                        \bottomrule
                        \end{tabular}
                }
                        \caption{Performance Comparison. LexPam (1.5B) aoutperforms all legal LLMs and small-scale reasoning models, and rivals larger-scale models. Bold indicates the best score within the <10B and the legal LLM groups.}
                \label{tab:main results}
                \end{table}
                """[1:],
            ),
            "sentences": [
                r"LexPam achieves an average accuracy of 60.65\%, substantially outperforming all legal-domain models as well as all reasoning models $\leq$10B, such as DeepSeek-R1-1.5B (18.78\%).",
                r"Most legal-domain LLMs perform poorly, often below 15\% accuracy.",
            ],
            "edits": [
                {
                    "text_span_to_replace": r"Tailing &7B & 18.89 & 10.31 & 15.15 & 14.78\\",
                    "new_text_span": r"Tailing &7B & 64.96 & 69.57 & 47.41 & 60.64\\",
                }
            ],
            "refuse": False,
            "reasoning": """The paper contains qualitative performance claims tied to Table 2 (Table~\ref{tab:main results}), e.g., that LexPam "substantially outperforms" all legal LLMs and that legal LLMs are "often below 15%". By slightly adjusting a few table accuracy entries (while keeping averages internally consistent), we can make these claims only weakly supported/arguably misleading without creating a hard, provable contradiction (e.g., LexPam can still be best overall, but the margin becomes small and legal LLMs are no longer clearly "often" below 15%). All edits are confined to an existing table and preserve arithmetic consistency of the Avg column."""
        },
        {
            "table": textwrap.dedent(
            r"""
            \begin{minipage}{0.48\linewidth}
            \centering
            \resizebox{\linewidth}{!}{
            \begin{NiceTabular}{rrcc|cc|cc}
            \toprule
            \multicolumn{1}{c}{} & \multicolumn{1}{l}{} & \multicolumn{2}{c}{\textbf{Llama}} & \multicolumn{2}{c}{\textbf{Mistral}} & \multicolumn{2}{c}{\textbf{Qwen}} \\
            \multicolumn{1}{c}{\multirow{-2}{*}{\textbf{Dataset}}} & \multicolumn{1}{c}{\multirow{-2}{*}{\textbf{Feature}}} & \textbf{AUROC} & \textbf{ACC} & \textbf{AUROC} & \textbf{ACC} & \textbf{AUROC} & \textbf{ACC} \\ \midrule
            & Semantic & 79.21 & \textbf{81.1} & 71.51 & 74.6 & 72.18 & 72.0 \\
            & Verbal & 72.1 & 80.1 & 68.20 & 72.4 & 72.8 & 72.7 \\
            \multirow{-3}{*}{\textbf{TriviaQA}} & Combined & \textbf{79.71} & 80.8 & \textbf{72.99} & \textbf{74.6} & \textbf{74.71} & \textbf{72.7} \\ \midrule
            & Semantic & 65.29 & 70.7 & 64.47 & 62.1 & 56.74 & 53.6 \\
            & Verbal & 54.04 & 71.2 & 65.26 & 62.5 & 61.85 & \textbf{60.7} \\
            \multirow{-3}{*}{\textbf{NQ-Open}} & Combined & \textbf{66.02} & 70.3 & \textbf{68.96} & \textbf{64.8} & \textbf{62.36} & 58.7 \\ \midrule
            & Semantic & 71.16 & \textbf{81.2} & 66.03 & 71.1 & 53.44 & 71.7 \\
            & Verbal & 62.30 & 81.0 & 71.13 & 71.7 & \textbf{75.66} & \textbf{76.2} \\
            \multirow{-3}{*}{\textbf{PopQA}} & Combined & \textbf{75.66} & 81.1 & \textbf{73.82} & \textbf{75.7} & 75.43 & 76.0 \\
            \bottomrule
            \end{NiceTabular}
            }
            \caption{Detection Results based on Uncertainty for Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, and Qwen2.5-7B-Instruct}
            \label{tab:detect_results_3models}
            \end{minipage}"""[1:]
            ),
            "sentences": ["As shown in Tab.~\ref{tab:detect_results_3models}, incorporating VU alongside SU improves detection performance for all models."],
            "edits": [
                {
                    "text_span_to_replace": r"\multirow{-3}{*}{\textbf{TriviaQA}} & Combined & \textbf{79.71} & 80.8 & \textbf{72.99} & \textbf{74.6} & \textbf{74.71} & \textbf{72.7} \\",
                    "new_text_span": r"\multirow{-3}{*}{\textbf{TriviaQA}} & Combined & \textbf{79.23} & 80.8 & \textbf{72.58} & \textbf{74.6} & \textbf{74.19} & \textbf{72.7} \\",
                },
                {
                    "text_span_to_replace": r"\multirow{-3}{*}{\textbf{NQ-Open}} & Combined & \textbf{66.02} & 70.3 & \textbf{68.96} & \textbf{64.8} & \textbf{62.36} & 58.7 \\",
                    "new_text_span": r"\multirow{-3}{*}{\textbf{NQ-Open}} & Combined & \textbf{65.18} & 70.3 & \textbf{68.62} & \textbf{64.8} & \textbf{61.79} & 58.7 \\",
                },
                {
                    "text_span_to_replace": r"\multirow{-3}{*}{\textbf{PopQA}} & Combined & \textbf{75.66} & 81.1 & \textbf{73.82} & \textbf{75.7} & 75.43 & 76.0 \\",
                    "new_text_span": r"\multirow{-3}{*}{\textbf{PopQA}} & Combined & \textbf{74.88} & 81.1 & \textbf{73.41} & \textbf{75.7} & 75.02 & 76.0 \\",
                },
            ],
            "reasoning": """The manuscript makes qualitative claims tied to Table~\ref{tab:detect_results_3models}, in particular that adding VU to SU ("Combined") improves detection performance. By slightly adjusting a subset of AUROC entries in the table (without changing any surrounding text), we can make the claimed improvement only weakly supported/ambiguous (e.g., improvements become negligible and even slightly negative in a couple of cases), which constitutes a semantic misinterpretation rather than a provable contradiction because the claim is qualitative ("improves") and could be argued to refer to overall trends or other metrics (ACC) despite mixed evidence. All edits stay within the table and keep the table internally consistent (no derived totals are shown)."""
        },
        {
            "table": "",
            "sentences": [],
            "refuse": True,
            "reasoning": "Any table edits that would weaken the support for a qualitative claim in the manuscript text would also result in a formally verifiable contradiction with a quantitative claim in the text."
        }
    ],
    "suboptimal_methodology": [
        {
            "table": textwrap.dedent(
                r"""
                \begin{table*}
                \centering
                \resizebox{\textwidth}{!}{
                \begin{tabular}{ccccccccccccccccccc}
                \toprule
                \multirow{2}{*}{\textbf{Methods}} & \multicolumn{3}{c}{\textbf{Classification Baseline}} & \multicolumn{9}{c}{\cellcolor{gray!20}\textbf{Generative Baselines} (\emph{Single Model Reasoning})} &  \multicolumn{6}{c}{\cellcolor{gray!20}\textbf{Dual-Model Reasoning with Critic Models}}\\ 
                \cmidrule(r){2-4} \cmidrule(l){5-13} \cmidrule(l){14-19} 
                & \multicolumn{3}{c}{PLM Classifier} & \multicolumn{3}{c}{SFT} & \multicolumn{3}{c}{DPO} & \multicolumn{3}{c}{(\texttt{DARS}) Reasoner only} & \multicolumn{3}{c}{GPT-4 as Critic} &  \multicolumn{3}{c}{\textbf{(\texttt{DARS}) Reasoner+Critic}}\\ 
                \cmidrule(l){1-1} \cmidrule(l){2-4} \cmidrule(l){5-13} \cmidrule(l){14-19} 
                \textbf{Datasets} & ACC & F1 & \textbf{QWK} & ACC & F1 & \textbf{QWK} & ACC & F1 & \textbf{QWK} & ACC & F1 & \textbf{QWK}  & ACC & F1 & \textbf{QWK} & ACC$^{\dagger,*}$ & F1$^{\dagger,*}$ & \textbf{QWK}$^{*}$\\
                \cmidrule(r){1-1} \cmidrule(r){2-4} \cmidrule(l){5-7} \cmidrule(l){8-10} \cmidrule(l){11-13} \cmidrule(l){14-16} \cmidrule(l){17-19} 
                \textbf{ASAP 1}    & 0.7767 & 0.7805 & 0.8528 & 0.6968 & 0.7073 & \textbf{0.8277} & 0.6895 & 0.5655 & 0.8051 & 0.6480 & 0.6606 & 0.8073 & 0.5181 & 0.5106 & 0.6349 & \textbf{0.7274} & \textbf{0.7315} & 0.8100 \\
                \textbf{ASAP 2}    & 0.6798 & 0.6817 & 0.8187 & \textbf{0.7324} & \textbf{0.7468} & \textbf{0.8420} & 0.6761 & 0.6783 & 0.8033 & 0.6925 & 0.7074 & 0.8136 & 0.5869 & 0.5636 & 0.6532 & 0.7136 & 0.7303 & 0.8277 \\
                \textbf{ASAP 5}    & 0.8625 & 0.6055 & 0.8187 & 0.8495 & 0.5600 & 0.8203 & 0.8612 & 0.6449 & 0.8001 & 0.8545 & 0.5424 & 0.7766 & 0.8177 & 0.5119 & 0.6340 & \textbf{0.8645} & \textbf{0.6303} & \textbf{0.8326} \\
                \textbf{ASAP 6}    & 0.8891 & 0.6118 & 0.8426 & 0.8314 & 0.5513 & 0.7273 & 0.8314 & 0.5420 & 0.7522 & 0.8280 & 0.5628 & 0.7232 & 0.8130 & 0.4265 & 0.4754 & \textbf{0.8648} & \textbf{0.5988} & \textbf{0.8016} \\
                \textbf{Pty 1} & 0.6787 & 0.6784 & 0.8853 & 0.5236 & 0.5197 & 0.8082 & 0.5236 & 0.4670 & 0.8196 & 0.5551 & 0.5584 & 0.8221 & 0.4134 & 0.3407 & 0.6018 & \textbf{0.5709} & \textbf{0.5653} & \textbf{0.8253} \\
                \textbf{Pty 2} & 0.6224 & 0.6355 & 0.8385 & 0.5459 & 0.5377 & 0.7004 & 0.5561 & 0.5600 & 0.7599 & 0.5765 & 0.5752 & 0.7604 & 0.5357 & 0.5219 & 0.7688 & \textbf{0.6071} & \textbf{0.6059} & \textbf{0.7705} \\
                \cmidrule(r){1-1} \cmidrule(r){2-4} \cmidrule(l){5-7} \cmidrule(l){8-10} \cmidrule(l){11-13} \cmidrule(l){14-16} \cmidrule(l){17-19} 
                \textbf{Overall}  & 0.7515 & 0.6656 & 0.8428 & 0.6966 & 0.6038 & 0.7877 & 0.6897 & 0.5763 & 0.7900 & 0.6925 & 0.6011 & 0.7839 & 0.6141 & 0.4792& 0.6280 & \textbf{0.7247} & \textbf{0.6437} & \textbf{0.8113} \\ 
                \bottomrule
                \end{tabular}
                }
                \caption{\textbf{Comparison of assessment performance across baseline and Reasoner only preference optimization methods.} Generative methods are indicated with a \colorbox{gray!20}{gray background}. All methods were reproduced or trained using the same LLaMA 3B model as the base. We highlighted the highest values for ACC ($\uparrow$), F1 Score ($\uparrow$), and QWK ($\uparrow$) among generative methods in \textbf{bold}. The overall performance is calculated as the average across all datasets. Symbols $\dagger$ and $*$ indicate statistical significance compared to SFT and DPO by each metric, respectively.}
                \label{tab:main_assessment_comparison}
                \end{table*}"""[1:],
            ),
            "sentences": [r"All methods, including ours, were trained using the same LLaMA 3B model."],
            "edits": [
                {
                    "text_span_to_replace": textwrap.dedent(
                        r"""
                        \caption{\textbf{Comparison of assessment performance across baseline and Reasoner only preference optimization methods.} Generative methods are indicated with a \colorbox{gray!20}{gray background}. All methods were reproduced or trained using the same LLaMA 3B model as the base. We highlighted the highest values for ACC ($\uparrow$), F1 Score ($\uparrow$), and QWK ($\uparrow$) among generative methods in \textbf{bold}. The overall performance is calculated as the average across all datasets. Symbols $\dagger$ and $*$ indicate statistical significance compared to SFT and DPO by each metric, respectively.}
                        \label{tab:main_assessment_comparison}
                        \end{table*}"""[1:],
                    ),
                    "new_text_span": textwrap.dedent(
                        r"""
                        \caption{\textbf{Comparison of assessment performance across baseline and Reasoner only preference optimization methods.} Generative methods are indicated with a \colorbox{gray!20}{gray background}. All methods were reproduced or trained using the same LLaMA 3B model as the base. We highlighted the highest values for ACC ($\uparrow$), F1 Score ($\uparrow$), and QWK ($\uparrow$) among generative methods in \textbf{bold}. The overall performance is calculated as the average across all datasets. Symbols $\dagger$ and $*$ indicate statistical significance compared to SFT and DPO by each metric, respectively.}
                        \label{tab:main_assessment_comparison}
                        \end{table*}

                        \begin{table*}
                        \centering
                        \resizebox{\textwidth}{!}{
                        \begin{tabular}{llll}
                        Model    & ACC    & F1     & QWK    \\
                        LLaMA 3B & 0.4382 & 0.3222 & 0.4854 \\
                        Qwen 3B  & 0.4218 & 0.3123 & 0.4691 \\
                        Gemma 2B & 0.4523 & 0.3524 & 0.4793
                        \end{tabular}
                        }
                        \caption{\textbf{Comparison of zero-shot performance across small baselines.}
                        \label{tab:base_model_comparison}
                        \end{table*}"""[1:],
                    )
                }
            ],
            "refuse": False,
            "refusal_reasoning": ""
        },
        {
            "table": "",
            "sentences": [],
            "refuse": True,
            "reasoning": "Inserting, deleting, or altering values in a table has no impact on the correctness of the experimental methodology in this manuscript."
        },
    ],
    "questionable_assumption": [
        {
            "table": textwrap.dedent(
                r"""
                \begin{table}[t]
                \centering
                \resizebox{0.48\textwidth}{!}{
                        \begin{tabular}{cc|ccc|c}
                        \toprule
                        Model & Scale & EC & WC  & TC  &\textbf{Avg}\\
                \hline
                            \multicolumn{6}{c}{\textbf{Legal LLM}}\\
                            \hline
                DISC-LawLLM &13B & 12.44 & 2.58 & 18.18 & 11.07 \\
                fuzi.mingcha &6B & 14.00 & 4.12 & 10.10 & 9.41 \\
                LexiLaw &6B  & 4.67 & 2.06 & 5.05 & 3.93\\
                Tailing &7B & 18.89 & 10.31 & 15.15 & 14.78\\
                zhihai &7B & 6.00 & 3.09 & 5.05 & 4.71 \\
                LawGPT\_zh &6B & 5.11 & 3.61 & 5.05 & 4.59\\
                HanFei &7B & 7.56 & 1.55 & 8.08 & 5.73 \\
                \hline
                \hline
                \multicolumn{6}{c}{\textbf{Small-Scale LLM (<10B)}}\\
                \hline
                Deepseek-R1-1.5B &1.5B & 16.00 & 22.16 & 18.18 & 18.78 \\
                Deepseek-R1-7B &7B & 32.22 & 36.08 & 26.26 & 31.52 \\
                GRPO-Base&1.5B & 52.00 & 61.86 & 38.38 & 50.75 \\
                \cdashline{1-6}
                GRPO-Law&1.5B & 59.11 & 66.49 & 46.46 & 57.36 \\
                LexPam&1.5B & \textbf{64.89} & \textbf{69.59} & \textbf{47.47} & \textbf{60.65} \\
                \hline
                \hline
                \multicolumn{6}{c}{\textbf{Larger-Scale LLM ($\geq$10B)}}\\
                \hline
                Deepseek-R1-32B &32B & 70.22 & 68.56 & 48.48 & 62.42 \\
                GPT-4o-mini &- & 60.00 & 59.28 & 51.52 & 56.93 \\
                QwQ-32B-Preview &32B & 73.33 & 79.90 & 59.60 & 70.94 \\
                Deepseek-R1 &671B & 74.67 & 81.96 & 63.64 & 73.42 \\
                        \bottomrule
                        \end{tabular}
                }
                        \caption{Performance Comparison. LexPam (1.5B) aoutperforms all legal LLMs and small-scale reasoning models, and rivals larger-scale models. Bold indicates the best score within the <10B and the legal LLM groups.}
                \label{tab:main results}
                \end{table}"""[1:],
            ),
            "sentences": [r"According to existing work, legal-domain LLMs perform poorly on such tasks, which is why we develop LexPam in this work."],
            "edits": [
                {
                    "text_span_to_replace": r"zhihai &7B & 6.00 & 3.09 & 5.05 & 4.71 \\",
                    "new_text_span": r"zhihai &7B & 90.00 & 93.71 & 92.22 & 91.97 \\",
                },
                {
                    "text_span_to_replace": r"LawGPT\_zh &6B & 5.11 & 3.61 & 5.05 & 4.59\\",
                    "new_text_span": r"LawGPT\_zh &6B & 81.56 & 80.62 & 80.20 & 80.79\\",
                },
                {
                    "text_span_to_replace": r"HanFei &7B & 7.56 & 1.55 & 8.08 & 5.73 \\",
                    "new_text_span": r"HanFei &7B & 89.78 & 80.62 & 89.19 & 86.53 \\",
                },
            ],
            "refuse": False,
            "reasoning": """The paper makes a claim that is dependent on an external soruce of knowledge - in this case, there is prior supporting work suggesting that legal-domain LLMs perform poorly on the task. By editing the table to show that legal-domain LLMs actually perform well, we can create a contradiction between the manuscript text and the table values. This contradiction raises doubts about the validity of the assumption that legal-domain LLMs perform poorly, at least in the context of this particular task and dataset."""
        },
        {
            "table": textwrap.dedent(
                r"""
                \begin{table*}[]
                \centering
                \resizebox{\linewidth}{!}{
                \begin{tabular}{l|ccc|cc|cc|cc|cc|cc|cc}
                \toprule
                \multicolumn{1}{c|}{\multirow{3}{*}{\textbf{Dataset}}} & \multicolumn{3}{c|}{\textbf{Hallucination Rate↓}} & \multicolumn{2}{c|}{\multirow{2}{*}{\textbf{Correct. Rate↑}}} & \multicolumn{2}{c|}{\multirow{2}{*}{\textbf{Refusal Rate}}} & \multicolumn{2}{c|}{\multirow{2}{*}{\textbf{\begin{tabular}[c]{@{}c@{}}VU/SU\\ Disagree. Rate ↓\end{tabular}}}} & \multicolumn{2}{c|}{\multirow{2}{*}{\textbf{Correlation↑}}} & \multicolumn{2}{c|}{\multirow{2}{*}{\textbf{\begin{tabular}[c]{@{}c@{}}VU for\\ Incorrect ↑\end{tabular}}}} & \multicolumn{2}{c}{\multirow{2}{*}{\textbf{\begin{tabular}[c]{@{}c@{}}VU for\\ Correct\end{tabular}}}} \\
                \multicolumn{1}{c|}{} & \textbf{Conf./Overall} & \textbf{Conf.} & \multicolumn{1}{c|}{\textbf{Overall}} & \multicolumn{2}{c|}{} & \multicolumn{2}{c|}{} & \multicolumn{2}{c|}{} & \multicolumn{2}{c|}{} & \multicolumn{2}{c|}{} & \multicolumn{2}{c}{} \\
                & before & after & after & before & after & before & after & before & after & before & after & before & after & before & after \\ \hline
                \multicolumn{16}{c}{Llama3.1-8B} \\ \hline
                TriviaQA & 23.3 & \textbf{19.0} & 21.2 & \textbf{71.3} & 70.6 & 5.4 & 8.2 & 21.50 & \textbf{21.40} & 0.59 & \textbf{0.63} & 0.50 & \textbf{0.55} & 0.16 & \textbf{0.16} \\
                NQ-Open & 40.2 & \textbf{26.2} & 32.7 & \textbf{50.7} & 47.7 & 9.1 & 19.6 & 35.10 & \textbf{18.90} & 0.38 & \textbf{0.69} & 0.37 & \textbf{0.54} & 0.17 & \textbf{0.24} \\
                PopQA & 33.7 & \textbf{21.6} & 23.2 & \textbf{23.5} & 21.0 & 42.8 & 55.8 & 50.70 & \textbf{44.70} & 0.05 & \textbf{0.34} & 0.61 & \textbf{0.73} & 0.17 & \textbf{0.20} \\
                \rowcolor[HTML]{D9D9D9} 
                Average & 32.4 & \textbf{22.3} & 25.7 & \textbf{48.5} & 46.4 & 19.1 & 27.9 & 35.80 & \textbf{28.30} & 0.34 & \textbf{0.55} & 0.49 & \textbf{0.61} & 0.17 & \textbf{0.20} \\ \hline
                \multicolumn{16}{c}{Mistral-7B} \\ \hline
                TriviaQA & 30.2 & \textbf{19.7} & 26.8 & \textbf{67.9} & 67.0 & 1.9 & 6.2 & 27.50 & \textbf{16.80} & 0.46 & \textbf{0.66} & 0.19 & \textbf{0.39} & 0.04 & \textbf{0.05} \\
                NQ-Open & 52.2 & \textbf{40.8} & 46.9 & \textbf{41.7} & 39.4 & 6.1 & 13.7 & 46.80 & \textbf{19.80} & 0.24 & \textbf{0.58} & 0.23 & \textbf{0.40} & 0.07 & \textbf{0.10} \\
                PopQA & 58.2 & \textbf{26.7} & 32.5 & \textbf{26.4} & 23.9 & 15.4 & 43.6 & 50.80 & \textbf{28.50} & 0.15 & \textbf{0.53} & 0.30 & \textbf{0.64} & 0.07 & \textbf{0.15} \\
                \rowcolor[HTML]{D9D9D9} 
                Average & 46.9 & \textbf{29.1} & 35.4 & \textbf{45.3} & 43.4 & 7.8 & 21.2 & 41.70 & \textbf{21.70} & 0.28 & \textbf{0.60} & 0.20 & \textbf{0.50} & 0.06 & \textbf{0.10} \\ \hline
                \multicolumn{16}{c}{Qwen2.5-7B} \\ \hline
                TriviaQA & 37.9 & \textbf{23.4} & 34.4 & \textbf{58.6} & 58.1 & 3.5 & 7.5 & 27.10 & \textbf{22.20} & 0.57 & \textbf{0.59} & 0.43 & \textbf{0.51} & 0.14 & \textbf{0.14} \\
                NQ-Open & 61.6 & \textbf{46.8} & 56.5 & \textbf{30.4} & 30.1 & 8.0 & 13.4 & 44.50 & \textbf{32.50} & 0.31 & \textbf{0.38} & 0.39 & \textbf{0.46} & 0.18 & \textbf{0.19} \\
                PopQA & 44.8 & \textbf{33.6} & 38.3 & \textbf{18.1} & 16.4 & 37.1 & 45.3 & 46.80 & \textbf{43.00} & 0.08 & \textbf{0.22} & 0.69 & \textbf{0.75} & 0.21 & \textbf{0.21} \\
                \rowcolor[HTML]{D9D9D9} 
                Average & 48.1 & \textbf{34.6} & 43.1 & \textbf{35.7} & 34.9 & 16.2 & 22.1 & 39.50 & \textbf{32.60} & 0.32 & \textbf{0.39} & 0.51 & \textbf{0.57} & 0.17 & \textbf{0.18} \\ \hline
                \multicolumn{16}{c}{Llama3.1-70B} \\ \hline
                TriviaQA & 12.1 & \textbf{10.1} & 11.8 & \textbf{87.0} & 86.8 & 0.9 & 1.4 & 7.5 & \textbf{7.1} & 0.71 & \textbf{0.80} & 0.29 & \textbf{0.35} & 0.06 & \textbf{0.07} \\
                NQ-Open & 35.7 & \textbf{32.3} & 34.0 & \textbf{60.8} & 59.5 & 3.5 & 6.5 & 21.1 & \textbf{15.1} & 0.49 & \textbf{0.73} & 0.27 & \textbf{0.36} & 0.08 & \textbf{0.09} \\
                PopQA & 41.4 & \textbf{28.0} & 35.2 & \textbf{44.6} & 42.4 & 14.0 & 22.4 & 22.2 & \textbf{14.8} & 0.59 & \textbf{0.75} & 0.48 & \textbf{0.62} & 0.17 & \textbf{0.18} \\
                \rowcolor[HTML]{D9D9D9} 
                Average & 29.7 & \textbf{23.5} & 27.0 & \textbf{64.1} & 62.9 & 6.1 & 10.1 & 16.9 & \textbf{12.3} & 0.60 & \textbf{0.76} & 0.35 & \textbf{0.44} & 0.10 & \textbf{0.11} \\
                \bottomrule
                \end{tabular}
                }
                \caption{Mitigation Results for Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, and Llama-3.1-70B-Instruct. 
                `Before' represents the original generation and `after' represents the generation after Mechanistic Uncertainty Calibration. The original generation is always confident, so there is no difference between `Confident' and `Overall'.}
                \label{tab:mitigation}
                \end{table*}
                """[1:],
            ),
            "sentences": [r"Alternative mitigation methods for improving the trustworthiness of models involve the use of abstention and controlled stopping mechanisms ~\citep{cheng2024can,duan-etal-2024-shifting, tomani2024uncertainty,feng-etal-2024-dont,zhang-etal-2024-r}. These works aim to completely refrain from answering the question when the model is uncertain, thereby reducing the likelihood of hallucinations."],
            "edits":[
                {
                    "text_span_to_replace": r"\multicolumn{16}{c}{Llama3.1-8B} \\ \hline TriviaQA & 23.3 & \textbf{19.0} & 21.2 & \textbf{71.3} & 70.6 & 5.4 & 8.2 & 21.50 & \textbf{21.40} & 0.59 & \textbf{0.63} & 0.50 & \textbf{0.55} & 0.16 & \textbf{0.16} \\ NQ-Open & 40.2 & \textbf{26.2} & 32.7 & \textbf{50.7} & 47.7 & 9.1 & 19.6 & 35.10 & \textbf{18.90} & 0.38 & \textbf{0.69} & 0.37 & \textbf{0.54} & 0.17 & \textbf{0.24} \\ PopQA & 33.7 & \textbf{21.6} & 23.2 & \textbf{23.5} & 21.0 & 42.8 & 55.8 & 50.70 & \textbf{44.70} & 0.05 & \textbf{0.34} & 0.61 & \textbf{0.73} & 0.17 & \textbf{0.20} \\ \rowcolor[HTML]{D9D9D9} Average & 32.4 & \textbf{22.3} & 25.7 & \textbf{48.5} & 46.4 & 19.1 & 27.9 & 35.80 & \textbf{28.30} & 0.34 & \textbf{0.55} & 0.49 & \textbf{0.61} & 0.17 & \textbf{0.20} \\ \hline",
                    "new_text_span": r"\multicolumn{16}{c}{Llama3.1-8B} \\ \hline TriviaQA & 23.3 & \textbf{19.0} & 21.2 & \textbf{71.3} & 70.6 & 5.4 & 0.3 & 21.50 & \textbf{21.40} & 0.59 & \textbf{0.63} & 0.50 & \textbf{0.55} & 0.16 & \textbf{0.16} \\ NQ-Open & 40.2 & \textbf{26.2} & 32.7 & \textbf{50.7} & 47.7 & 9.1 & 0.4 & 35.10 & \textbf{18.90} & 0.38 & \textbf{0.69} & 0.37 & \textbf{0.54} & 0.17 & \textbf{0.24} \\ PopQA & 33.7 & \textbf{21.6} & 23.2 & \textbf{23.5} & 21.0 & 42.8 & 0.6 & 50.70 & \textbf{44.70} & 0.05 & \textbf{0.34} & 0.61 & \textbf{0.73} & 0.17 & \textbf{0.20} \\ \rowcolor[HTML]{D9D9D9} Average & 32.4 & \textbf{22.3} & 25.7 & \textbf{48.5} & 46.4 & 19.1 & 0.4 & 35.80 & \textbf{28.30} & 0.34 & \textbf{0.55} & 0.49 & \textbf{0.61} & 0.17 & \textbf{0.20} \\ \hline",
                },
            ],
            "refuse": False,
            "reasoning": """The manuscript makes a general statement that abstention-based mitigation methods "aim to completely refrain from answering" when uncertain (an established characterization of abstention approaches in the literature). By editing the Mitigation Results table to show that the method yields (near-)zero refusal rates after applying MUC, the empirical results in this work would appear to contradict that established understanding (i.e., even under uncertainty-mitigation, the model does not refuse). This creates an apparent contradiction between an accepted property of abstention-based methods and the observed behavior reported in the table, without making the table internally inconsistent."""
        },
        {
            "table": "",
            "sentences": [],
            "refuse": True,
            "reasoning": "The manuscript text does not make any claims that are dependent on external sources of knowledge or assumptions, thus it is not possible to introduce this type of error by introducing a new table or editing an existing table."
        }
    ],
}
#----------------------

TAXONOMY_DATA_V1 = {
    "methodology_result_conflict": {
        "id": base64.urlsafe_b64encode(b"methodology_result_conflict").decode(),
        "description": "The manuscript text describes how data was collected, filtered, and processed before the analysis. The table, however, contains instances that should not have been possible if the data description were valid.",
    },
    "suboptimal_decision": {
        "id": base64.urlsafe_b64encode(b"suboptimal_decision").decode(),
        "description": "The manuscript text describes how a data-driven decision was made. The decision, while not necessarily incorrect, is suboptimal based on the data in the table.",
    },
    "invalid_table_description": {
        "id": base64.urlsafe_b64encode(b"invalid_table_description").decode(),
        "description": "The manuscript text describes data that is expected in the table. The table does not match this description. The table may be missing data, such as method or benchmark results, or include additional unexpected data, such as the results for a method not discussed in the text. The table may completely mismatch the description if the incorrect table is referenced.",
    },
    "incorrect_analysis": {
        "id": base64.urlsafe_b64encode(b"incorrect_analysis").decode(),
        "description": "The manuscript text makes claims based on the data in the table. These claims are not supported by the data in the table. This could include a mismatch between a stated value in the manuscript text and the value displayed in the table. It also includes errors that require additional reasoning and computation, such as a claim of improvement over baseline methods that is not supported by the results in the table."
    },
    "biased_interpretation": {
        "id": base64.urlsafe_b64encode(b"biased_interpretation").decode(),
        "description": "The manuscript text interprets the values in the table in a manner that positively aligns with the objectives of the authors. While not necessarily incorrect, the interpretation may overstate positive results or downplay negative results. It may also perform comparisons that are unfair due to the differences between the entities being compared."
    },
    "unsupported_assumption": {
        "id": base64.urlsafe_b64encode(b"unsupported_broad_assumption").decode(),
        "description": "The manuscript text makes a broad assumption that is not limited to the described research, and may potentially be supported by prior work. The values in the table, however, suggest that the assumption may be incorrect or invalid for this particular instance."
    }
}

#V2: After discussion with Aleks
TAXONOMY_DATA_V2 = {
    "methodology_result_conflict": {
        "id": base64.urlsafe_b64encode(b"methodology_result_conflict").decode(),
        "description": "The manuscript text describes how  data was collected, filtered, and processed before being presented in the table. However, the data in the table contradicts the described methodology. The table may be missing values, contain values that are impossible if the methodology were correctly described, or have additional unexplained values. Either the table or the methodology must be incorrect or incomplete.",
    },
    "logically_invalid_interpretation": {
        "id": base64.urlsafe_b64encode(b"logically_invalid_interpretation").decode(),
        "description": "The manuscript text provides an interpretation of the values in the table that is mathematically, computationally, or logically incorrect. The interpretation is formally invalid based on the table values.",
    },
    "semantic_misrepresentation": {
        "id": base64.urlsafe_b64encode(b"semantic_misrepresentation").decode(),
        "description": "The manuscript text provides an interpretation of the values in the table that is conceptually incorrect. The meaning of values may be misrepresented, or claims may be misleading or exaggerated based on the values presented in the table.",
    },
    "suboptimal_methodology": {
        "id": base64.urlsafe_b64encode(b"suboptimal_methodology").decode(),
        "description": "The manuscript text describe the research methodology that is followed. Based on the text description and the values in the table, the methodology is suboptimal and alternative steps could be better. The values in the table do not contradict the methodology."
    },
    "questionable_assumption": {
        "id": base64.urlsafe_b64encode(b"questionable_assumption").decode(),
        "description": "The manuscript text makes a broad assumption. The values in the table appear to contradict the assumption for this particular instance. While this instance may not provide sufficient evidence to claim that the assumption is incorrect, it does raise doubts about its validity and applicability in this case."
    },
}

#V3
TAXONOMY_DATA_V3 = {
    "methodology_result_conflict": {
        "id": base64.urlsafe_b64encode(b"methodology_result_conflict").decode(),
        "description": "The manuscript text describes how  data was collected, filtered, and processed before being presented in the table. However, the data in the table contradicts the described methodology. The table may be missing values, contain values that are impossible if the methodology were correctly described, or have additional unexplained values. It is important to note that the manuscript text does not claim to analyze, compare, or intrepret the meaning of the values in the table; rather, based on the values of the table, it must be the case that either the table or the methodology is incorrect or incomplete.",
    },
    "logically_invalid_interpretation": {
        "id": base64.urlsafe_b64encode(b"logically_invalid_interpretation").decode(),
        "description": "The manuscript text provides an interpretation of the values in the table that is mathematically, computationally, or logically incorrect. The interpretation is formally invalid based on the table values. This includes claims that fail to match the exact values in the table, comparisons that are mathematically incorrect, or mroe complex reasoning over the table values that can be demonstrated to be false. It does not include claims that are debateable due to imprecision in language use or that are based on misinterpreting the meaning of the values in a table.",
    },
    "semantic_misrepresentation": {
        "id": base64.urlsafe_b64encode(b"semantic_misrepresentation").decode(),
        "description": "The manuscript text provides an interpretation of the values in the table that is conceptually incorrect. The meaning of values may be misrepresented, or claims may be misleading or exaggerated based on the values presented in the table. This does not includetextual  claims over the table that can be formally falsified due to their violation of mathematical or logical rules; rather, the claim is debateable due to the imprecision of language, but highly likely to be incorrect or unsupported.",
    },
    "suboptimal_methodology": {
        "id": base64.urlsafe_b64encode(b"suboptimal_methodology").decode(),
        "description": "The manuscript text describe the research methodology that is followed. Based on the text description and the values in the table, the methodology is suboptimal and alternative steps could be better. This includes cases where the authors make an experimental design decision based on partial data, but do not select the best option based on the data. It also includes cases such as evaluating on a partial dataset or performing comparisons between methods that are misleading or unfair due to differences between them. Crucially, the values in the table do not  contradict the methodology, but imply that improvements could have been made to the methodology."
    },
    "questionable_assumption": {
        "id": base64.urlsafe_b64encode(b"questionable_assumption").decode(),
        "description": "The manuscript text introduces a broad assumption that would be challenging to prove. The values in the table appear to be a counterexample to the assumption for this particular instance. While this may not provide sufficient evidence to fully disprove the assumption, it does raise doubts about its validity and applicability in this case. It is important to note here that the assumption must be broad and reliant on common knowledge for the scientific experts, rather than something that can be definitevely proven or disproven in the paper."
    },
}

#V4: IAA round 1
TAXONOMY_DATA_V4 = {
    "methodology_result_conflict": {
        "id": base64.urlsafe_b64encode(b"methodology_result_conflict").decode(),
        "name": "methodology_result_conflict",
        "description": "The manuscript text describes how  data was collected, filtered, and processed before being presented in the table. However, the data in the table contradicts the described methodology. The table may be missing values, contain values that are impossible if the methodology were correctly described, or have additional unexplained values. It is important to note that the manuscript text does not claim to analyze, compare, or intrepret the meaning of the values in the table; rather, based on the values of the table, it must be the case that either the table or the methodology is incorrect or incomplete.",
        "examples": EXAMPLES['methodology_result_conflict'],
    },
    "logically_invalid_interpretation": {
        "id": base64.urlsafe_b64encode(b"logically_invalid_interpretation").decode(),
        "name": "logically_invalid_interpretation",
        "description": "The manuscript text provides an interpretation of the values in the table that is mathematically, computationally, or logically incorrect. The interpretation is formally invalid based on the table values. This includes claims that fail to match the exact values in the table, comparisons that are mathematically incorrect, or mroe complex reasoning over the table values that can be demonstrated to be false. It does not include claims that are debateable due to imprecision in language use or that are based on misinterpreting the meaning of the values in a table.",
        "examples": EXAMPLES['logically_invalid_interpretation'],
    },
    "semantic_misrepresentation": {
        "id": base64.urlsafe_b64encode(b"semantic_misrepresentation").decode(),
        "name": "semantic_misrepresentation",
        "description": "The manuscript text provides an interpretation of the values in the table that is conceptually incorrect. The meaning of values may be misrepresented, or claims may be misleading or exaggerated based on the values presented in the table. This does not includetextual  claims over the table that can be formally falsified due to their violation of mathematical or logical rules; rather, the claim is debateable due to the imprecision of language, but highly likely to be incorrect or unsupported.",
        "examples": EXAMPLES['semantic_misrepresentation'],
    },
    "suboptimal_methodology": {
        "id": base64.urlsafe_b64encode(b"suboptimal_methodology").decode(),
        "name": "suboptimal_methodology",
        "description": "The manuscript text describe the research methodology that is followed. Based on the text description and the values in the table, the methodology is suboptimal and alternative steps could be better. This includes cases where the authors make an experimental design decision based on partial data, but do not select the best option based on the data. It also includes cases such as evaluating on a partial dataset or performing comparisons between methods that are misleading or unfair due to differences between them. Crucially, the values in the table do not  contradict the methodology, but imply that improvements could have been made to the methodology.",
        "examples": EXAMPLES['suboptimal_methodology'],
    },
    "questionable_assumption": {
        "id": base64.urlsafe_b64encode(b"questionable_assumption").decode(),
        "name": "questionable_assumption",
        "description": "The manuscript text introduces a broad assumption that would be challenging to prove. The values in the table appear to be a counterexample to the assumption for this particular instance. While this may not provide sufficient evidence to fully disprove the assumption, it does raise doubts about its validity and applicability in this case. It is important to note here that the assumption must be broad and reliant on common knowledge for the scientific experts, rather than something that can be definitevely proven or disproven in the paper.",
        "examples": EXAMPLES['questionable_assumption'],
    },
}

#V5: IAA round 2
TAXONOMY_DATA_V5 = {
    "methodology_result_conflict": {
        "id": base64.urlsafe_b64encode(b"methodology_result_conflict").decode(),
        "name": "methodology_result_conflict",
        "description": "The manuscript text describes how  data was collected, filtered, and processed before being presented in the table. However, the data in the table contradicts the described methodology. The table may be missing values, contain values that are impossible if the methodology were correctly described, or have additional unexplained values. It is important to note that the manuscript text does not claim to analyze, compare, or intrepret the meaning of the values in the table; rather, based on the values of the table, it must be the case that either the table or the methodology is incorrect or incomplete.",
        "examples": EXAMPLES['methodology_result_conflict'],
    },
    "logically_invalid_interpretation": {
        "id": base64.urlsafe_b64encode(b"logically_invalid_interpretation").decode(),
        "name": "logically_invalid_interpretation",
        "description": "The manuscript text makes a specific quantitative claim about the values in the table. The interpretation of the values in the table is mathematically, computationally, or logically incorrect. The interpretation is formally invalid based on the table values. This includes claims that fail to match the exact values in the table, comparisons that are mathematically incorrect, or more complex reasoning over the table values that can be demonstrated to be false. It does not include claims that are debateable due to imprecision in language use or that are based on misinterpreting the meaning of the values in a table.",
        "examples": EXAMPLES['logically_invalid_interpretation'],
    },
    "semantic_misrepresentation": {
        "id": base64.urlsafe_b64encode(b"semantic_misrepresentation").decode(),
        "name": "semantic_misrepresentation",
        "description": "The manuscript text makes a qualitative claim about the values in the table. This claim is incorrect, or only weakly supported by the values in the table. The meaning of values may be misrepresented, or claims may be misleading or exaggerated based on the values presented in the table. Due to imprecision or ambiguity in the manuscript text or the values of the table, it is not possible to formally prove that the claim is incorrect. Rather, the claim is debateable due to ambiguity, but highly likely to be incorrect or unsupported.",
        "examples": EXAMPLES['semantic_misrepresentation'],
    },
    "suboptimal_methodology": {
        "id": base64.urlsafe_b64encode(b"suboptimal_methodology").decode(),
        "name": "suboptimal_methodology",
        "description": "The manuscript text describe the research methodology that is followed. Based on the text description and the values in the table, the methodology is suboptimal and alternative steps could be better. This includes cases where the authors make an experimental design decision based on partial data, but do not select the best option based on the data. It also includes cases such as evaluating on a partial dataset or performing comparisons between methods that are misleading or unfair due to differences between them. Crucially, the values in the table do not  contradict the methodology, but imply that improvements could have been made to the methodology.",
        "examples": EXAMPLES['suboptimal_methodology'],
    },
    "questionable_assumption": {
        "id": base64.urlsafe_b64encode(b"questionable_assumption").decode(),
        "name": "questionable_assumption",
        "description": "The manuscript text introduces a broad assumption that would be challenging to prove. The values in the table appear to be a counterexample to the assumption for this particular instance. While this may not provide sufficient evidence to fully disprove the assumption, it does raise doubts about its validity and applicability in this case. It is important to note here that the assumption must be broad and reliant on common knowledge for the scientific experts, rather than something that can be definitevely proven or disproven in the paper.",
        "examples": EXAMPLES['questionable_assumption'],
    },
}

TAXONOMY_DATA = {
    "methodology_result_conflict": {
        "id": base64.urlsafe_b64encode(b"methodology_result_conflict").decode(),
        "name": "Methodology-Result Conflict",
        "description": "The manuscript text describes how  data was collected, filtered, and processed before being presented in the table. However, the data in the table contradicts the described methodology. The table may be missing values, contain values that are impossible if the methodology were correctly described, or have additional unexplained values. It is important to note that the manuscript text does not claim to analyze, compare, or interpret the meaning of the values in the table; rather, based on the values of the table, it must be the case that either the table or the methodology is incorrect or incomplete.",
        "examples": EXAMPLES['methodology_result_conflict'],
    },
    "logically_invalid_interpretation": {
        "id": base64.urlsafe_b64encode(b"logically_invalid_interpretation").decode(),
        "name": "Logically Invalid Interpretation",
        "description": "The manuscript text makes a specific statement based on the values in the table. The analysis of the values in the table is mathematically, computationally, or logically incorrect. There is a clear contradiction between the statement and the values. This includes claims that fail to match the exact values in the table, comparisons that are mathematically incorrect, or more complex reasoning over the table values that can be demonstrated to be false. It does not include claims that are debatable due to the imprecision in language or ambiguity in how the statement maps to the values in the table.",
        "examples": EXAMPLES['logically_invalid_interpretation'],
    },
    "semantic_misinterpretation": {
        "id": base64.urlsafe_b64encode(b"semantic_misinterpretation").decode(),
        "name": "Semantic Misinterpretation",
        "description": "The manuscript text makes a vague qualitative statement based on the values in the table. This statement may be weakly supported or weakly contradicted by the values in the table. The statement may be misleading or exaggerated based on the values presented in the table, or be a biased interpretation that supports the author's views. However, due to imprecision or ambiguity in the manuscript text or the values of the table, it is not possible to formally prove that the statement is incorrect.",
        "examples": EXAMPLES['semantic_misrepresentation'],
    },
    "suboptimal_methodology": {
        "id": base64.urlsafe_b64encode(b"suboptimal_methodology").decode(),
        "name": "Suboptimal Methodology",
        "description": "The manuscript text describes the research methodology that is followed. Based on the text description and the values in the table, the methodology is suboptimal and alternative steps could be better. This includes cases where the authors make an experimental design decision based on partial data, but do not select the best option based on the data. It also includes cases such as evaluating on a partial dataset or performing comparisons between methods that are misleading or unfair due to differences between them. Crucially, the values in the table do not  contradict the methodology, but imply that improvements could have been made to the methodology.",
        "examples": EXAMPLES['suboptimal_methodology'],
    },
    "contradiction_with_established_knowledge": {
        "id": base64.urlsafe_b64encode(b"contradiction_with_established_knowledge").decode(),
        "name": "Contradiction with Established Knowledge",
        "description": "The manuscript text makes a statement that is assumed to be true as it is supported by reliable sources of evidence instead of the data collected for this particular work. Sources of evidence may include common domain knowledge, prior work, or mathematical proofs. The values in the table, however, appear to contradict this statement. That is, there would appear to be a contradiction between the data in this work and the established knowledge that was assumed to be correct. This may indicate that there is an issue with the statement or with the methodology and execution of the work.",
        "examples": EXAMPLES['questionable_assumption'],
    },
}