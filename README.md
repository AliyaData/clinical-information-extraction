\# Clinical Information Extraction Pipeline



\## AgenticX Generative AI \& LLM Engineering Internship – Brief 4



A structured clinical information extraction pipeline that converts unstructured clinical text into validated JSON containing:



\- Conditions

\- Drugs

\- Dosages

\- Procedures



The project uses a typed Pydantic schema, an instruction-tuned language model, structured-output validation, controlled retry handling, and quantitative evaluation on 100 clinical text samples.



\---



\## 1. Project Objective



Clinical notes contain important medical information in an unstructured form. Extracting this information manually can be time-consuming and difficult to standardize.



This project builds an automated pipeline that takes clinical text as input and produces structured information in a predefined JSON format.



\### Input



Unstructured clinical text.



Example:



> Patient was diagnosed with diabetes and was prescribed Metformin 500 mg twice daily.



\### Output



```json

{

&#x20; "conditions": \[

&#x20;   "diabetes"

&#x20; ],

&#x20; "drugs": \[

&#x20;   "Metformin"

&#x20; ],

&#x20; "dosages": \[

&#x20;   "500 mg twice daily"

&#x20; ],

&#x20; "procedures": \[]

}

