#!/bin/bash

# ### JAVA Version
# # gpt4, document-level, toplevel
# model="in1B-poly-codellama7B-star2_3B"
# machine="gpt4"
# mode="toplevel"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/toplevel/java/codenet\(JAVA\)-human-machine-toplevel-${machine}.jsonl \
#   --output_file /KSY_copy/dataset/toplevel/java/feature_extraction/${model}/codenet\(JAVA\)_${machine}_${mode}_line_features.jsonl

# # gemini, document-level, toplevel
# model="in1B-poly-codellama7B-star2_3B"
# machine="gemini"
# mode="toplevel"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/toplevel/java/codenet\(JAVA\)-human-machine-toplevel-${machine}.jsonl \
#   --output_file /KSY_copy/dataset/toplevel/java/feature_extraction/${model}/codenet\(JAVA\)_${machine}_${mode}_line_features.jsonl

# # gpt4, line-level, toplevel
# model="in1B-poly-codellama7B-star2_3B"
# machine="gpt4"
# mode="toplevel"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/toplevel/java/codenet\(JAVA\)_gpt4o_mini_toplevel_line_level.jsonl \
#   --output_file /KSY_copy/dataset/toplevel/java/feature_extraction/${model}/codenet\(JAVA\)_${machine}_${mode}_hybrid_line_features.jsonl

# # gemini, line-level, toplevel
# model="in1B-poly-codellama7B-star2_3B"
# machine="gemini"
# mode="toplevel"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/toplevel/java/codenet\(JAVA\)_gemini_2.0_flashlite_toplevel_line_level.jsonl \
#   --output_file /KSY_copy/dataset/toplevel/java/feature_extraction/${model}/codenet\(JAVA\)_${machine}_${mode}_hybrid_line_features.jsonl



# ### C++ Version
# # gpt4, document-level, toplevel
# model="in1B-poly-codellama7B-star2_3B"
# machine="gpt4"
# mode="toplevel"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/toplevel/cpp/codenet\(cpp\)-human-machine-toplevel-${machine}.jsonl \
#   --output_file /KSY_copy/dataset/toplevel/cpp/feature_extraction/${model}/codenet\(cpp\)_${machine}_${mode}_line_features.jsonl


# # gemini, document-level, toplevel
# model="in1B-poly-codellama7B-star2_3B"
# machine="gemini"
# mode="toplevel"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/toplevel/cpp/codenet\(cpp\)-human-machine-toplevel-${machine}.jsonl \
#   --output_file /KSY_copy/dataset/toplevel/cpp/feature_extraction/${model}/codenet\(cpp\)_${machine}_${mode}_line_features.jsonl


# # gpt4, line-level, toplevel
# model="in1B-poly-codellama7B-star2_3B"
# machine="gpt4"
# mode="toplevel"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/toplevel/cpp/codenet\(cpp\)_gpt4o_mini_toplevel_line_level.jsonl \
#   --output_file /KSY_copy/dataset/toplevel/cpp/feature_extraction/${model}/codenet\(cpp\)_${machine}_${mode}_hybrid_line_features.jsonl

# # gemini, line-level, toplevel
# model="in1B-poly-codellama7B-star2_3B"
# machine="gemini"
# mode="toplevel"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/toplevel/cpp/codenet\(cpp\)_gpt4o_mini_toplevel_line_level.jsonl \
#   --output_file /KSY_copy/dataset/toplevel/cpp/feature_extraction/${model}/codenet\(cpp\)_${machine}_${mode}_hybrid_line_features.jsonl


## Python Version


# model="in1B-poly-codellama7B-star2_3B"
# machine="gpt4"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/codenet\(python\)-human-machine-${machine}_executable_final.jsonl \
#   --output_file /KSY_copy/dataset/feature_extraction/${model}/codenet\(python\)_${machine}_line_features_executable_final.jsonl

# model="in1B-poly-codellama7B-star2_3B"
# machine="gemini"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/codenet\(python\)-human-machine-${machine}_executable_final.jsonl \
#   --output_file /KSY_copy/dataset/feature_extraction/${model}/codenet\(python\)_${machine}_line_features_executable_final.jsonl


## 새 데이터셋 space 처리한 버전
# model="in1B-poly-codellama7B-star2_3B"
# machine="gpt4"
# mode="toplevel"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/toplevel/python/python_recompile/codenet\(python\)-human-machine-toplevel-${machine}_space.jsonl \
#   --output_file /KSY_copy/dataset/toplevel/python/feature_extraction/${model}/codenet\(python\)_${machine}_${mode}_line_features_space.jsonl


# model="in1B-poly-codellama7B-star2_3B"
# machine="gemini"
# mode="toplevel"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/toplevel/python/python_recompile/codenet\(python\)-human-machine-toplevel-${machine}_space.jsonl \
#   --output_file /KSY_copy/dataset/toplevel/python/feature_extraction/${model}/codenet\(python\)_${machine}_${mode}_line_features_space.jsonl

# ## 기존 데이터셋 space 처리한 버전
# model="in1B-poly-codellama7B-star2_3B"
# machine="gpt4"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/codenet-human-machine-dataset-${machine}_space.jsonl \
#   --output_file /KSY_copy/dataset/feature_extraction/${model}/codenet\(python\)_${machine}_line_features_space.jsonl


# model="in1B-poly-codellama7B-star2_3B"
# machine="gemini"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/codenet-human-machine-dataset-${machine}_space.jsonl \
#   --output_file /KSY_copy/dataset/feature_extraction/${model}/codenet\(python\)_${machine}_line_features_space.jsonl

# model="in1B-poly-codellama7B-star2_3B"
# machine="gemini"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/codenet-human-machine-dataset-${machine}_single_per_problem.jsonl \
#   --output_file /KSY_copy/dataset/feature_extraction/${model}/codenet\(python\)_${machine}_line_features_single_per_problem.jsonl

# model="in1B-poly-codellama7B-star2_3B"
# machine="gpt4"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/codenet-human-machine-dataset-${machine}_single_per_problem.jsonl \
#   --output_file /KSY_copy/dataset/feature_extraction/${model}/codenet\(python\)_${machine}_line_features_single_per_problem.jsonl


# gpt4, line-level, toplevel
# model="in1B-poly-codellama7B-star2_3B"
# machine="gpt4"
# mode="toplevel"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/toplevel/python/codenet\(python\)_gpt4o_mini_toplevel_line_level.jsonl \
#   --output_file /KSY_copy/dataset/toplevel/python/feature_extraction/${model}/codenet\(python\)_${machine}_${mode}_hybrid_line_features.jsonl

# ## gemini, line-level, toplevel
# model="in1B-poly-codellama7B-star2_3B"
# machine="gemini"
# mode="toplevel"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/toplevel/python/codenet\(python\)_gemini_2.0_flashlite_toplevel_line_level.jsonl \
#   --output_file /KSY_copy/dataset/toplevel/python/feature_extraction/${model}/codenet\(python\)_${machine}_${mode}_hybrid_line_features.jsonl

# # gpt4, document-level, toplevel
# model="in1B-poly-codellama7B-star2_3B"
# machine="gpt4"
# mode="toplevel"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/toplevel/python/codenet\(python\)-human-machine-toplevel-${machine}.jsonl \
#   --output_file /KSY_copy/dataset/toplevel/python/feature_extraction/${model}/codenet\(python\)_${machine}_${mode}_line_features.jsonl

# # gemini, document-level, toplevel
# model="in1B-poly-codellama7B-star2_3B"
# machine="gemini"
# mode="toplevel"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/toplevel/python/codenet\(python\)-human-machine-toplevel-${machine}.jsonl \
#   --output_file /KSY_copy/dataset/toplevel/python/feature_extraction/${model}/codenet\(python\)_${machine}_${mode}_line_features.jsonl


### Python Old Version
# gpt4, line-level, toplevel
# model="in1B-poly-codellama7B-star2_3B"
# machine="gpt4"
# mode="toplevel"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/toplevel/python/codenet\(python\)_gpt4o_mini_toplevel_line_level.jsonl \
#   --output_file /KSY_copy/dataset/toplevel/python/feature_extraction/${model}/codenet\(python\)_${machine}_${mode}_hybrid_line_features.jsonl

# model="in1B-poly-codellama7B-star2_3B"
# machine="gemini"
# mode="toplevel"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/toplevel/python/codenet\(python\)_gpt4o_mini_toplevel_line_level.jsonl \
#   --output_file /KSY_copy/dataset/toplevel/python/feature_extraction/${model}/codenet\(python\)_${machine}_${mode}_hybrid_line_features.jsonl

# # gpt4, line-level, toplevel - balanced samples
# model="in1B-poly-codellama7B-star2_3B"
# machine="gpt4"
# mode="toplevel"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/toplevel/python/python_single_sample/codenet\(python\)_gpt4o_mini_toplevel_line_level_sample.jsonl \
#   --output_file /KSY_copy/dataset/toplevel/python/python_single_sample/feature_extraction/${model}/codenet\(python\)_${machine}_${mode}_hybrid_line_features.jsonl

# ## gemini, line-level, toplevel - balanced samples
# model="in1B-poly-codellama7B-star2_3B"
# machine="gemini"
# mode="toplevel"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/toplevel/python/python_single_sample/codenet\(python\)_gemini_2.0_flashlite_toplevel_line_level_sample.jsonl \
#   --output_file /KSY_copy/dataset/toplevel/python/python_single_sample/feature_extraction/${model}/codenet\(python\)_${machine}_${mode}_hybrid_line_features.jsonl

# single_sample_per_problem
# model="in1B-poly-codellama7B-star2_3B"
# machine="gemini"
# mode="toplevel"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/toplevel/python/codenet\(python\)-human-machine-toplevel-${machine}_single_per_problem_deduplicated.jsonl \
#   --output_file /KSY_copy/dataset/toplevel/python/python_single_sample/feature_extraction/${model}/codenet\(python\)_${machine}_${mode}_line_features_deduplicated.jsonl

# model="in1B-poly-codellama7B-star2_3B"
# machine="gpt4"
# mode="toplevel"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/toplevel/python/codenet\(python\)-human-machine-toplevel-${machine}_single_per_problem_deduplicated.jsonl \
#   --output_file /KSY_copy/dataset/toplevel/python/python_single_sample/feature_extraction/${model}/codenet\(python\)_${machine}_${mode}_line_features_deduplicated.jsonl

# model="in1B-poly-codellama7B-star2_3B"
# machine="gemini"
# mode="toplevel"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/toplevel/python/codenet\(python\)-human-machine-toplevel-${machine}_single_per_problem_deduplicated_no_def_solve.jsonl \
#   --output_file /KSY_copy/dataset/toplevel/python/python_single_sample/feature_extraction/${model}/codenet\(python\)_${machine}_${mode}_line_features_deduplicated_no_def_solve.jsonl



# model="in1B-poly-codellama7B-star2_3B"
# machine="gemini"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/toplevel/python/aigcodeset/codenet\(python\)_aigcodeset_generate_only.jsonl \
#   --output_file /KSY_copy/dataset/toplevel/python/aigcodeset/feature_extraction/${model}/codenet\(python\)_${machine}_line_features_aigcodeset.jsonl

# model="in1B-poly-codellama7B-star2_3B"
# machine="gemini"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/toplevel/python/aigcodeset/codenet\(python\)_python_document_level_merged_file_0820.jsonl \
#   --output_file /KSY_copy/dataset/toplevel/python/aigcodeset/feature_extraction/${model}/codenet\(python\)_${machine}_line_features_all_models_new.jsonl

# model="in1B-poly-codellama7B-star2_3B"
# machine="gemini"
# python3 gen_features.py \
#   --input_file /KSY_copy/dataset/toplevel/python/aigcodeset/codenet\(python\)_aigcodeset_gemini_only.jsonl \
#   --output_file /KSY_copy/dataset/toplevel/python/aigcodeset/feature_extraction/${model}/codenet\(python\)_${machine}_line_features_gpt4o.jsonl

model="in1B-poly-codellama7B-star2_3B"
machine="gemini"
python3 gen_features.py \
  --input_file /KSY_copy/dataset/toplevel/cpp/aigcodeset/codenet\(cpp\)_aigcodeset_line_level_merged_0916.jsonl \
  --output_file /KSY_copy/dataset/toplevel/cpp/aigcodeset/feature_extraction/${model}/codenet\(cpp\)_${machine}_line_features_hybrid.jsonl
