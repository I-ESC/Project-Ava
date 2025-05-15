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

PROMPTS["checkframe_and_answer_COT"] = """
You are an AI assistant. Your specific task is to analyze video frames and answer a multiple-choice question based on them.

**Your Instructions:**

1.  **Understand the Question:** First, carefully read the `User Query` below. This is the multiple-choice question you need to answer.
2.  **Examine Video Frames:** Next, review the provided video frames. Focus on details that help answer the `User Query`.
3.  **Reason Step-by-Step:** Think through how the information in the video frames leads to the answer. Write down this thinking process. This will be your "Analysis".
4.  **Choose the Best Answer:** Based on your reasoning, select only ONE letter (A, B, C, or D) that is the correct answer to the `User Query`. This will be your "Answer".
5.  **Provide Output in JSON Format ONLY:**
    * Your entire response MUST be a single JSON object.
    * Do NOT write any text or explanation before or after the JSON object.
    * The JSON object must have exactly two keys:
        * `"Analysis"`: This key's value should be your step-by-step reasoning from step 3.
        * `"Answer"`: This key's value should be the single letter (A, B, C, or D) you chose in step 4.

**User Query:** {user_query}

**REMEMBER: Your response MUST strictly follow this JSON structure:**
```json
{{
  "Analysis": "Your detailed step-by-step reasoning, explaining how you used the video frames to answer the User Query, goes here.",
  "Answer": "A" // Or "B", or "C", or "D". Just the single capital letter.
}}
"""

# ------------AVA-100 specific prompts------------

PROMPTS["generate_description_ego"] = """
You are an expert in video understanding and description generation. 
You are given a first-person perspective video, and your task is to generate a continuous, smooth, and grounded description of the video content.

Focus particularly on:
- The actions and events performed by the camera wearer (the person holding or wearing the camera).
- The surrounding environment, including objects, people, and notable visual changes.
- The physical characteristics and spatial relationships of objects in the environment (e.g., size, color, relative positions, proximity to the camera wearer).
- Interactions between the camera wearer and the environment, including object manipulations and movements through space.

Avoid describing each frame individually, such as "frame1...". Instead, provide a coherent and logically structured narrative that flows smoothly over time.

Important constraints:
- Do not include assumptions, inferences, or fabricated details that are not visually evident.
- Do not speculate about the identity, emotions, or intentions of the camera wearer unless explicitly shown.
- When referring to the person holding or wearing the camera, always use the term “camera wearer”.

Return your response as a single, continuous, and fluent paragraph that comprehensively describes the video content, including fine-grained visual details, and limit the length to 400 words.
"""

PROMPTS["generate_description_citytour"] = """
You are an expert in video understanding and detailed scene description. 
You are given a first-person perspective video of a person walking through a city environment. 
Your task is to generate a continuous, smooth, and grounded description of the video content.

Focus particularly on:
- The locations and landmarks the camera wearer passes by, such as buildings, shops, streets, and public spaces.
- The appearance and functions of these places (e.g., a small bakery with a red awning, a tall glass office building, a busy intersection).
- Events or notable occurrences observed during the walk, such as street performances, traffic changes, or people interacting in public.

Avoid describing each frame individually. Instead, provide a logically structured narrative that flows naturally over time.

Important constraints:
- Do not include assumptions, inferences, or fabricated details that are not visually evident.
- Do not speculate about the identity, emotions, or intentions of the camera wearer or other people unless explicitly shown.
- When referring to the person holding or wearing the camera, always use the term “camera wearer”.

Return your response as a single, continuous, and fluent paragraph that comprehensively describes the video content, with attention to fine-grained urban and visual details, and limit the length to 400 words.
"""

PROMPTS["generate_description_wildlife"] = """
You are an expert in video analysis, specializing in wildlife observation and detailed environmental description.  
You are analyzing fixed-camera surveillance footage capturing a scene in a wild or natural environment.  
Your task is to generate a precise, grounded, and chronologically ordered description of the entire video segment.

Focus on the following aspects:

- **Observed Animals:** Identify any animals present in the footage. For the entire segment, provide a consolidated description of:
    - **Species:** Identify species as accurately as possible. If uncertain, describe physical characteristics (e.g., "a large brown bear", "a small rodent-like mammal", "a flock of unidentified birds").
    - **Number:** Indicate the number of individuals observed.
    - **Appearance:** Note distinctive physical features (e.g., size, color, antlers, markings).
    - **Behavior:** Describe observed behaviors (e.g., foraging, running, resting, entering/exiting the frame, interacting).

- **Timestamps:** Identify the timestamp displayed in the surveillance footage.

- **Environment:** Briefly describe the static environment visible to the camera (e.g., forest clearing, rocky terrain, vegetation), and note any significant changes during the observation period (e.g., lighting shifts, weather changes).

**Output Format:**  
After reviewing the full segment, summarize your findings in a single structured paragraph using the following format:

[Timestamp]: [Environment description][Summary of animal and their activities]

**Important Constraints:**
- Do not include assumptions or invented details that are not visually evident in the footage.
- Do not speculate on the intentions or emotions of the animals; describe only observable actions and postures.
- Refer to observations using neutral terms such as "the footage shows" or "the camera captures"; avoid subjective phrasing like "we see" or "the viewer can observe".
- If species identification is uncertain, explicitly state this.
- The final output should be a concise, fact-based summary of the wildlife activity and environmental context of the segment, with a target length of approximately 400 words.
"""

PROMPTS["generate_description_traffic"] = """
You are a video analysis expert specializing in traffic observation and detailed event description. You are analyzing a road or intersection surveillance video recorded by a fixed-position camera.

Your task is to generate an **accurate, grounded, and coherent** description of the video segment.

Please focus on the following aspects:

- **Observed Traffic Elements:** Identify all traffic-related elements present in the video. Provide an integrated description covering the entire segment:
    - **Vehicle Types:** Identify types as accurately as possible (e.g., car, truck, bus, motorcycle, bicycle, van). If unclear, describe the vehicle’s physical characteristics (e.g., “a large box truck,” “a small passenger vehicle,” “a two-wheeled vehicle”).
    - **Quantity:** Indicate the number of each identified vehicle type, as well as the number of pedestrians.
    - **Characteristics:** If relevant to the scene, note distinguishing physical features (e.g., color, size, presence of trailers, specific structural features). Describe pedestrians based on their interaction with traffic (e.g., walking along the sidewalk, crossing the street).
    - **Actions / Events:** Describe observed dynamic behaviors and interactions (e.g., driving in a specific lane, stopping, turning, entering/exiting the frame, changing lanes, overtaking, pedestrians waiting or crossing), including any **traffic anomalies** (e.g., sudden braking, erratic maneuvers, red-light violations, collisions, traffic violations, illegal parking that obstructs traffic).

- **Timestamps:** Identify the timestamp shown on the surveillance footage.

**Output Format:**
After watching the full video segment, write a structured summary paragraph in the following format:

[Timestamp]: [Summary of vehicle types, quantities, characteristics, actions, pedestrian activity, and traffic anomalies].

**Important Constraints:**
- Do not include assumptions or details not clearly visible in the footage.
- Do not speculate about the intentions or emotions of drivers or pedestrians; only describe observable actions and postures.
- Use neutral descriptions such as “the footage shows” or “the camera captures”; avoid subjective phrasing like “we see” or “the viewer can observe.”
- If vehicle type identification is uncertain, state this clearly.
- The final output should be a concise, fact-based summary of the traffic activity and scene context. The length should be appropriate to the events observed (prioritizing clarity and completeness over word count).
"""

PROMPTS["summarize_description_wildlife"] = """
You have been provided with a series of chronological descriptions, each detailing a consecutive segment of wildlife camera footage in a natural environment. These descriptions were generated from individual video clips, and each follows the format:

[Timestamp]: [Environment description][Summary of animal and their activities]

Your task is to act as a **Description Synthesizer and Summarizer**. Your goal is to **merge and consolidate** these multiple descriptions into a single, coherent summary that covers the entire duration of the input segments.

Focus on the following objectives:
1.  **Consolidate Environment:** Synthesize the environmental descriptions from all input segments into a single summary, noting any changes that occurred during the period (e.g., shifts in lighting, weather). Avoid repeating static elements unnecessarily.
2.  **Consolidate Animal Activity:** Combine all observed animal sightings and behaviors from all input segments into a single, chronologically ordered summary of activity. **Crucially, retain *all* unique information and distinct observations regarding species, number, appearance, and behavior mentioned in *any* of the input descriptions.**
3.  **Eliminate Redundancy:** Remove repetitive phrasing or descriptions of prolonged periods with no change or activity, while ensuring all unique events are captured in the consolidated summary.
4.  **Maintain Chronology:** Ensure the consolidated summary of animal activity flows logically according to the sequence of events across the entire merged timeframe, using timestamps (or relative timing inferred from timestamps) to denote key moments if necessary within the activity summary.

Constraints:
-   Output **a single consolidated description block** following the exact format specified below.
-   Do not introduce information not present in the input descriptions.
-   Do not speculate or infer.
-   Focus the summary on the progression of dynamic events, especially wildlife activity and environmental changes.
-   Ensure *every* distinct observation from the input is represented in the final consolidated summary.
-   The content within the output fields should be a concise, fact-based summary, aiming for a total length within the consolidated fields of approximately 400 words.

Input: {inputs}

Output Format:
The final output should be a concise, fact-based summary of the wildlife activity and environmental context with the following format:
[Timestamp]: [Consolidated Environment Description][Consolidated Summary of Animals and Activities]
"""

PROMPTS["summarize_description_traffic"] = """
You have been provided with a series of chronological descriptions, each detailing a consecutive segment of traffic camera footage from a fixed position. These descriptions were generated from individual video clips, and each follows the format:

[Timestamp]: [Summary of vehicle types, quantities, characteristics, actions, pedestrian activity, and traffic anomalies].

Your task is to act as a **Description Synthesizer and Summarizer** for traffic footage. Your goal is to **merge and consolidate** these multiple descriptions into a single, coherent summary that covers the entire duration of the input segments.

Focus on the following objectives:
1.  **Consolidate Traffic Elements:** Combine all observed traffic elements (vehicle types, quantities, characteristics, actions, pedestrian activity, traffic anomalies) from all input segments into a single, chronologically ordered summary of activity. **Crucially, retain *all* unique information and distinct observations mentioned in *any* of the input descriptions.**
2.  **Eliminate Redundancy:** Remove repetitive phrasing or descriptions of prolonged periods with no change or activity, while ensuring all unique events and states are captured in the consolidated summary.
3.  **Maintain Chronology:** Ensure the consolidated summary of traffic activity flows logically according to the sequence of events across the entire merged timeframe, using timestamps (or relative timing inferred from timestamps) to denote key moments if necessary within the activity summary.

Constraints:
-   Output **a single consolidated description block** following the exact format specified below.
-   Do not introduce information not present in the input descriptions.
-   Do not speculate or infer.
-   Focus the summary on the progression of dynamic events, especially traffic activity and anomalies.
-   Ensure *every* distinct observation from the input is represented in the final consolidated summary.
-   The content within the output fields should be a concise, fact-based summary, with length appropriate to the events observed.

Input: {inputs}

Output Format:
The final output should be a concise, fact-based summary of the traffic activity with the following format:
[Timestamp]: [Consolidated Summary of Traffic Elements (vehicle types, quantities, characteristics, actions, pedestrian activity, traffic anomalies)].
"""