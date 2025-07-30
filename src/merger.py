from pathlib import Path
import pandas as pd


class TicketDataMerger:
    def __init__(self, database_dir: Path):
        self.database_dir = database_dir

    def merge_results(
        self, categories_file: Path, summaries_file: Path, output_file: Path = None
    ) -> Path:
        """
        Combina os arquivos de categorias e resumos em um único arquivo CSV.
        """
        if output_file is None:
            output_file = self.database_dir / "final_analysis.csv"

        # Lê os arquivos
        categories_df = pd.read_csv(categories_file, sep=";")
        summaries_df = pd.read_csv(summaries_file, sep=";")

        # Faz o merge usando ticket_id como chave
        merged_df = pd.merge(categories_df, summaries_df, on="ticket_id", how="outer")

        # Salva o resultado
        merged_df.to_csv(output_file, sep=";", index=False, encoding="utf-8-sig")
        return output_file
