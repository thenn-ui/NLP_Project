from transformers import pipeline
import textwrap

summarizer = None

def init_summarizer():
    print("Initializing summarizer module...")
    global summarizer 
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def get_summary(text: str, max_chunk_chars: int = 2000, scale_factor: float = 0.25, depth=0, max_depth=2) -> str:

    global summarizer

    # basic summarizer --> ~2000 token limit on the bart summary model
    # if summarizer:
    #     result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    #     return result[0]['summary_text']
    # else:
    #     return "Summarizer not initialized. Call init_summarizer() first."

    # go with chunks. FIXED ONE PARAGRAPH SUMMARY

    # if summarizer is None:
    #     return "Summarizer not initialized"

    # # Step 1: Split text into chunks
    # chunks = textwrap.wrap(text, max_chunk_chars, break_long_words=False, break_on_hyphens=False)

    # # Step 2: Summarize each chunk
    # summaries = []
    # for i, chunk in enumerate(chunks):
    #     print(f"[INFO] Summarizing chunk {i+1}/{len(chunks)}...")
    #     result = summarizer(chunk, max_length=512, min_length=100, do_sample=False)
    #     summaries.append(result[0]['summary_text'])

    # # Step 3: Optionally summarize the combined summaries
    # if len(summaries) > 1:
    #     combined = " ".join(summaries)
    #     print(f"[INFO] Summarizing combined summary...")
    #     #final_summary = summarizer(combined, max_length=512, min_length=100, do_sample=False)[0]['summary_text']
    #     final_summary = get_summary(combined, max_chunk_chars, max_length, min_length)
    #     return final_summary

    # return summaries[0]

    if summarizer is None:
        return "Summarizer not initialized"

    # Step 1: Split text into chunks
    chunks = textwrap.wrap(text, max_chunk_chars, break_long_words=False, break_on_hyphens=False)
    summaries = []

    for i, chunk in enumerate(chunks):
        print(f"[INFO] Summarizing chunk {i+1}/{len(chunks)}...")
        
        chunk_len = len(chunk.split())  # word count
        max_length = max(int(chunk_len * scale_factor), 50)
        min_length = max(int(max_length * 0.6), 30)

        print(f"[INFO] chunk words: {chunk_len}, summary: min {min_length}, max {max_length}")
        
        result = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
        summaries.append(result[0]['summary_text'])

    # Recursive summarization of summaries (if needed)
    if len(summaries) > 1 and depth < max_depth:
        combined = " ".join(summaries)
        print(f"[INFO] Summarizing combined summary... (depth={depth + 1})")
        return get_summary(combined, max_chunk_chars, scale_factor, depth + 1, max_depth)

    return summaries[0]
