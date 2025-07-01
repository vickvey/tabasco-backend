# services/file_generation.py
from nltk.tokenize import word_tokenize

def generate_summary_files(target_word, clusters_dict, summary_folder_path, disamb_model=None):
    """
    Write summary files for each cluster showing related context words and representative sentences.
    """
    summary_folder_path.mkdir(parents=True, exist_ok=True)
    for cluster_num, sentences in clusters_dict.items():
        if disamb_model:
            # Use context words from disamb_model if available
            all_context_words = []
            for sentence in sentences:
                context_words = disamb_model.get_context_words(sentence, target_word, top_k=10)
                all_context_words.extend(context_words)
            word_sim_dict = {}
            for word, sim in all_context_words:
                word_sim_dict[word] = max(word_sim_dict.get(word, sim), sim)
            top_words = sorted(word_sim_dict.items(), key=lambda x: x[1], reverse=True)[:50]
            top_words_str = " , ".join([f"{word} ({sim:.4f})" for word, sim in top_words])
        else:
            # Fallback to word frequency if disamb_model not available
            word_freq = {}
            for sentence in sentences:
                tokens = word_tokenize(sentence)
                for word in tokens:
                    if word.lower() != target_word.lower() and word.isalpha():
                        word_freq[word] = word_freq.get(word, 0) + 1
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            top_words = [word for word, _ in sorted_words[:50]]
            top_words_str = " , ".join(top_words)

        file_path = summary_folder_path / f"summary_text_{cluster_num}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("Top-most context words from the cluster:\n")
            f.write("***********************************************\n")
            f.write(top_words_str + "\n")
            f.write("***********************************************\n\n")
            for idx, sent in enumerate(sentences[:5]):
                f.write(f"{target_word.title()} {idx+1}\n")
                f.write(f"Instance {idx+1} of {target_word.title()} belongs to Cluster {int(cluster_num)+1}\n")
                f.write("\n~~~~~~\n")
                f.write("Corresponding Sentence:\n")
                f.write(f"{sent}\n")
                f.write("##############################################################\n\n")

def generate_detailed_files(clusters, disamb_model, target_word, detailed_folder):
    """
    Write detailed files for each cluster showing context words for each sentence.
    """
    detailed_folder.mkdir(parents=True, exist_ok=True)
    for cluster_num, sentences in clusters.items():
        file_path = detailed_folder / f"text_{cluster_num}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            for idx, sentence in enumerate(sentences):
                context_words = disamb_model.get_context_words(sentence, target_word, top_k=10)
                f.write(f"{target_word.title()} {idx}\n")
                f.write(f"Instance {idx} of {target_word.title()} belongs to Cluster {cluster_num}\n")
                f.write("Corresponding Sentence:\n")
                f.write(f"{sentence}\n")
                f.write("Context Words:\n")
                for word, sim in context_words:
                    f.write(f"{word}: {sim:.4f}\n")
                f.write("--------------------------------------------------\n")
