PROMPTS = {}

PROMPTS["entity_relation_extraction"] = """
You are a vision-based information extraction expert. Your task is to analyze a sequence of frames and extract **key visible entities** and their **relationships**. Follow the instructions below step-by-step, and ensure the output strictly follows the specified JSON format.

---

### Step 1: Extract Visible Entities  
1. Identify **all distinct, visible, and detailed entities** in the frames, such as:
   - Humans, animals, objects, text elements or any other visually distinguishable entities.
   - If there are multiple instances of the same entity type, treat them as same entities.
2. Provide a **detailed description** of each entity in a concise and fluent manner, including:
   - Physical attributes (e.g., size, shape, color, texture).
   - Recognizable text content (if visible).
   - Relevant actions or interactions.
   - ohters
3. Record the **frame indices** where each entity is visible under the `Index` field.

---

### Step 2: Extract Relationships Between Entities  
1. Identify **clear and meaningful relationships** between TWO distinct entities listed IN Step 1.    
   - If a relationship introduces new entities, add those entities to the entities list but avoid duplication.
2. Provide a **concise, clear description** of the relationship, focusing only on **visual connections or interactions**. Avoid referencing their relationship to specific frames or time indices.


---

### Step 3: Return Structured Output  
Output the extracted information in the following **JSON format**. If no entities or relationships are found, return an empty list for the respective field:

```json
{
  "Entities": [
    {
      "Entity_name": "[ENTITY NAME]",
      "Entity_description": "[DETAILED DESCRIPTION OF VISUAL APPEARANCE]",
      "Index": [FRAME_INDICES]
    },
    ...
  ],
  "Relations": [
    {
      "Entity1": "[ENTITY NAME 1]",
      "Entity2": "[ENTITY NAME 2]",
      "Relation_description": "[CONCISE DESCRIPTION OF THE VISUAL CONNECTION OR INTERACTION]"
    },
    ...
  ]
}
"""

PROMPTS["generate_description"] = """
You are an expert in video understanding and description generation. 
Your task is to provide a continuous and smooth description of the video, focusing on the video content. Avoid describing each frame individually like "frame1 ...". 
The description should cover the main scenes, characters, objects, and any notable actions or changes, ensuring the description is coherent and logical. Finally, return your response as a single, continuous, and fluent paragraph that fully describes the video content and limit the length to 300 words.
"""

PROMPTS["summarize_descriptions"] = """
You are an expert in summarizing video segment descriptions. Your task is to extract segment information from the sequential video segment descriptions and merge them into a single video event description.

### Event Description Guidelines:
- If the content continues the same scene or content, MERGE then and DON'T duplicate the information.
- The tone of the event description should be as if you are directly describing the video event. Provide a comprehensive narrative of the merged events without separating the content into distinct segments or using line breaks to list different aspects.
- Refrain from using phrases such as "At 2.0 seconds...", "By 10.0 seconds...", "The first/last segment", "The second event begins with...", "The final frames of this segment" or ohter time-related words, which will make the event content fragmented.
- Only provide objective information, avoiding subjective interpretations like mood or atmosphere.

### Output Format:
Please provide the response in one continuous paragraph.

segment description in format of `start_time:end_time:description`:
{inputs} 
"""

PROMPTS["keyword_extraction"] = """
- Goal -
As a specialist in keyword extraction, your task is to identify and list the most relevant keywords from a given query. Focus on extracting keywords related to events and entities, avoiding terms specific to video or task context. These keywords should effectively capture the essence of the query to aid in accurate information retrieval. Present the keywords as a comma-separated list.

######################
- Examples -
######################

Question: Which animal does the protagonist encounter in the forest scene?
################
Output:
animal, protagonist, forest, encounter

Question: In the movie, what color is the car that chases the main character through the city?
################
Output:
color, car, chases, main character, city

Question: What is the weather like during the opening scene of the film?\n(A) Sunny\n(B) Rainy\n(C) Snowy\n(D) Windy
################
Output:
weather, opening scene, Sunny, Rainy, Snowy, Windy

#############################
- Real Data -
######################
Question: {input_text}
######################
Output: 
"""

PROMPTS["query_rewrite_for_entity_retrieval"] = """
- Goal -
For a given query, generate a declarative sentence to serve as a query for retrieving relevant knowledge, concentrating on the main entities and relevant descriptions.

######################
- Examples -
######################

Question: On a stage with lights, there are many people wearing colorful outfits. What are these people in the colorful outfits doing?
################
Output:
Stage with lights, people wearing colorful outfits

Question: What is special about the celebration in New York according to the video?\nA. Hosting large parades.\nB. Dressing in green and dyeing the river to green.\nC. Drinking a lot.\nD. Planting shamrocks.
################
Output:
New York, celebration, large parades, dyeing the river, drinking, planting shamrocks

Question: Which animals appear in the wildlife footage? \n(A) Lions\n(B) Elephants\n(C) Zebras
################
Output:
Animals that appear in the wildlife footage, lions, elephants, zebras

#############################
- Real Data -
######################
Question: {input_text}
######################
Output:
"""

PROMPTS["query_rewrite_for_visual_retrieval"] = """
- Goal -
Generate a declarative sentence to serve as a query for retrieving relevant video segments based on the provided question that may include scene-related information.

######################
- Examples -
######################

Question: Which animal does the protagonist encounter in the forest scene?
################
Output:
The protagonist encounters an animal in the forest.

Question: In the movie, what color is the car that chases the main character through the city?
################
Output:
A city chase scene where the main character is pursued by a car.

Question: What is the weather like during the opening scene of the film?\n(A) Sunny\n(B) Rainy\n(C) Snowy\n(D) Windy
################
Output:
The opening scene of the film featuring specific weather conditions. (Possibly Sunny, Rainy, Snowy, or Windy)

#############################
- Real Data -
######################
Question: {input_text}
######################
Output:
"""


PROMPTS["re-query"] = """
You are an advanced AI system tasked with generating a sub-query to retrieve new information based on the current query and the Information Retrieved from Videos. Your goal is to refine the search to obtain additional relevant data.

######################
- Instructions -
######################

1. Review the User Query and the Information Retrieved from Videos. Analyze how the information retrieved from videos can help answer the User Query.
2. Identify specific areas where additional information is insufficient to help answer the User Query.
3. According to the insufficient information, formulate a new query can help answer the User Query.
4. Directly output the new query in the field of "sub_query" with JSON format.

#############################
- Real Data -
######################
User Query: {user_query}

---Information Retrieved from Videos---
{video_segments}

######################
- Output -
######################
Response in JSON format:
{{
  "sub_query": "[SUB-QUERY]"
}}
"""

PROMPTS["summary_and_answer_COT"] = """
You are an advanced AI system designed to answer questions based on video content. When a user's query is presented, you will receive retrieved video segments, organized by timestamps and descriptions. Your task is to analyze these segments, synthesize the information, and select the best answer to the multiple-choice question based on the video. Respond with only the letter (A, B, C, or D). 

######################
- Instructions -
######################
1. Carefully review the provided video segment information, paying attention to timestamps and descriptions, and pick the most relevant information.
2. Conduct a detailed reasoning process to analyze the information and how they are related to the user query.
3. Select the best answer to the multiple-choice question based on the video. Respond with only the letter (A, B, C, or D). 
4. Return your review, reference and reasoning process in the `Analysis` field and the answer in the `Answer` field.

#############################
- Real Data -
######################
User Query: {user_query}

Video Segments (organized by timestamp, description):
{video_segments}

######################
- Output -
######################
Response in clean JSON format:
{{
  "Analysis": "[Analysis]",
  "Answer": "[A, B, C or D]"
}}
"""