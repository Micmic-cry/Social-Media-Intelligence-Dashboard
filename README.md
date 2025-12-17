# Social Media Analytics Dashboard

A Streamlit-based dashboard for analyzing social media performance, campaign effectiveness, and engagement metrics.

## Features

- **Executive Overview**: High-level performance metrics and trends over time.
- **ROI Analysis**: Evaluate campaign effectiveness and cost efficiency.
- **Platform Performance**: Compare performance across social media channels.
- **Geographic Analytics**: Visualize regional performance and engagement.
- **Data Explorer**: Explore raw data and export filtered datasets.

## Quick Start

1. **Install dependencies**: pip install streamlit pandas plotly numpy

2. **Prepare data**: Place `Cleaned_Viral_Social_Media_Trends.csv` in the same directory as the dashboard script.

3. **Run the dashboard**: streamlit run src/app.py (unless you are viewing this from the deployed huggingface site)


## Data Requirements

The dashboard expects a CSV file with the following columns: Platform, Region, Content_Type, Hashtag, Views, Likes, Shares, Comments, Post_Date


## Usage

- Navigate between analytical views using the sidebar.
- Apply filters by **Platform** and **Region** to refine the analysis.
- Download filtered datasets from the **Data Explorer** tab.
- Interactive charts support zooming, hovering, and detailed insights.

## Technical Notes

- Engagement metrics and ROI are automatically calculated.
- Outliers are handled to improve visualization clarity.
- Fully responsive design suitable for different screen sizes.
- All data processing is performed locally; no external services are required.




