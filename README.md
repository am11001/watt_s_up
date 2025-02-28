# TU Vienna Watt's Up Hackathon 
## London Energy Usage Analysis & Predictive Modeling
### Team: Ländle (Vorarlberg) meets Budapest
#### Purpose of this Project: 
<p> This project focuses on analyzing London's energy usage patterns in 2013 and developing predictive models using machine learning. The dataset contains large-scale time-series data of 8760 hours (1 full consecutive year). Our goal is to extract insights, categorize consumption patterns, and create synthetic data for future energy demand prediction while ensuring that individual user profiles remain unidentifiable in the synthetic dataset. </p>

#### Dataset: 
  <ul>
    <li> consists of timestamps (rows) and IDs (columns) representing different households. </li>
    <li> contains energy usage records but excludes personal information, as IDs are simply numbered. However, for the purpose of the hackathon, we still treat these IDs as personal data to ensure that the synthetic data cannot be inferred from the real data. </li>
    <li> due to its large size (~308MB), data is processed in chunks to improve efficiency. </li>
  </ul>

#### Technologies Used:
<ul>
  <li> Python: Data analysis and preprocessing </li>
  <li> Pandas, NumPy: Data handling and manipulation </li>
  <li> TensorFlow: Deep learning models </li>
  <li> Scikit-learn: Feature engineering and baseline models </li>
  <li> Matplotlib, Seaborn: Data visualization </li>
</ul>



