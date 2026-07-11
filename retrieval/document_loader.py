"""
文档加载器 — 加载 data/knowledge/ 目录下的文档并切分为文本块。

支持格式: .txt, .md, .pdf
分块策略:
- chunk_size=500: 每块最多 500 字符
- chunk_overlap=50: 相邻块重叠 50 字符，保持上下文连贯
"""

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# 知识库文档目录
KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge"


class DocumentLoader:
    """加载并切分知识库文档"""

    def __init__(self,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ".", "；", ";", " ", ""],
        )

    def load_file(self, file_path: str) -> list[Document]:
        """
        加载单个文件并切分。

        参数:
            file_path: 文件路径

        返回:
            Document 列表
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._load_pdf(path)
        else:
            return self._load_text(path)

    def load_directory(self, directory: str | None = None) -> list[Document]:
        """
        加载目录下所有支持的文件并切分。

        参数:
            directory: 目录路径，默认为 data/knowledge/

        返回:
            Document 列表
        """
        target_dir = Path(directory) if directory else KNOWLEDGE_DIR

        if not target_dir.exists():
            print(f"[DocumentLoader] 知识库目录不存在: {target_dir}")
            return []

        all_docs = []
        supported = {".txt", ".md", ".pdf"}

        for file_path in target_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported:
                try:
                    docs = self.load_file(str(file_path))
                    all_docs.extend(docs)
                    print(f"[DocumentLoader] 已加载: {file_path.name} → {len(docs)} 块")
                except Exception as e:
                    print(f"[DocumentLoader] 加载失败 {file_path.name}: {e}")

        print(f"[DocumentLoader] 总计加载 {len(all_docs)} 个文本块")
        return all_docs

    def _load_text(self, path: Path) -> list[Document]:
        """加载纯文本/Markdown 文件"""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # 添加元数据 (来源文件名)
        doc = Document(
            page_content=content,
            metadata={"source": path.name, "type": path.suffix},
        )
        return self.splitter.split_documents([doc])

    def _load_pdf(self, path: Path) -> list[Document]:
        """加载 PDF 文件 (需要 pypdf)"""
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(str(path))
            pages = loader.load()
            # 为每页标注来源
            for i, page in enumerate(pages):
                page.metadata["source"] = path.name
                page.metadata["page"] = i + 1
            return self.splitter.split_documents(pages)
        except ImportError:
            print("[DocumentLoader] pypdf 未安装，跳过 PDF 文件")
            return []
