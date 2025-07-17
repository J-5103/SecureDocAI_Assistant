# src/components/test_pipeline.py

import time
import difflib
from src.pipeline.question_answer_pipeline import QuestionAnswerPipeline
from src.pipeline.plot_pipeline import PlotGenerationPipeline
PDF_FILE = "test_files/sample.pdf"

# Expected answers (customize based on your PDF content!)
expected_answers = {
    "What is machine learning?": "Machine learning is a subset of AI that enables systems to learn from data.",
    "Give me the summary of the document": "This document contains machine learning concepts, algorithms, and examples."
}

# ⏳ Total start time
start_time = time.time()

# 1️⃣ Initialize Pipelines
qa_pipeline = QuestionAnswerPipeline()
# plot_pipeline = PlotGenerationPipeline()

correct = 0
total = 0

for question, expected in expected_answers.items():
    print(f"\n❓ Question: {question}")
    q_start = time.time()
    try:
        answer = qa_pipeline.run(question)
        print(f"✅ Answer: {answer}")

        # 🎯 Similarity using difflib
        similarity = difflib.SequenceMatcher(None, answer.lower(), expected.lower()).ratio()
        percentage = similarity * 100
        print(f"🔍 Similarity: {percentage:.2f}%")

        if percentage >= 80:
            print("✅ Correct Answer")
            correct += 1
        else:
            print("❌ Incorrect Answer")

        total += 1

    except Exception as e:
        print(f"❌ Error: {str(e)}")

    print(f"⏱️ Time Taken: {time.time() - q_start:.4f} sec")

# # 2️⃣ Plot Generation
# print("\n📊 Generating Plot...")

# try:
#     plot_path = plot_pipeline.generate_plot("Generate a bar chart of algorithms used in machine learning")
#     print(f"✅ Plot saved at: {plot_path}")
# except Exception as e:
#     print(f"❌ Plot generation failed: {str(e)}")

# 3️⃣ Report Accuracy
if total > 0:
    accuracy = (correct / total) * 100
    print(f"\n🎯 Accuracy: {accuracy:.2f}% ({correct}/{total})")

# ⏳ Total time
print(f"\n🚀 Test pipeline completed in {time.time() - start_time:.4f} sec")
