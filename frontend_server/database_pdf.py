import pandas as pd

df = pd.DataFrame({
    "name":["Anurag", "Anuu", "Kshitiz"],
    "email":["vanuragkumardesai@gmail.com", "gameranurag24@gmail.com", "anuragbusiness819@gmail.com"],
    "job_description":['''AI Engineer

Build and deploy AI/ML solutions, develop LLM-powered applications, optimize models, and collaborate with cross-functional teams. Requires Python, machine learning, deep learning frameworks, and cloud platform experience.''',

'''Full Stack Developer

Develop and maintain front-end and back-end applications, build APIs, manage databases, and ensure application performance and security. Requires JavaScript/TypeScript, React, Node.js, and database knowledge.''',

'''Data Analyst

Analyze data, create dashboards, generate insights, and support business decisions through reporting and visualization. Requires SQL, Excel, Power BI/Tableau, and strong analytical skills.''']
})

df.to_csv("job_csv.csv")