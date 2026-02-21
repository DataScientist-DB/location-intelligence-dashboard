Location Intelligence Dashboard
Spatial Market Attractiveness & Catchment Intelligence Platform
A professional-grade spatial analytics tool for evaluating retail expansion, service coverage, and competitive density within configurable catchment radii.
Designed for commercial use in:
•	Retail site selection
•	Pharmacy & healthcare expansion
•	Restaurant market analysis
•	Supermarket density benchmarking
•	Urban service accessibility assessment
•	Investment feasibility evaluation
 
🧠 Executive Intelligence Built In
The dashboard automatically generates:
•	📊 Market density per km²
•	🏁 Competitive pressure index
•	⚠ Entry risk score
•	📈 Opportunity percentage
•	📝 Executive recommendation
•	📄 Investor-ready PDF memo
This enables structured decision support rather than raw data exploration.
________________________________________
🔎 Core Capabilities
1️⃣ Multi-Radius Catchment Analysis
Evaluate performance within:
•	300m
•	500m
•	1000m
•	1500m
•	2000m
Density, competition, and opportunity are recalculated for each radius.
________________________________________
2️⃣ Competitive Pressure Modeling
The system computes:
Competitive Pressure = f(density, competitor share)
Entry Risk = f(quality, competition, saturation)
Opportunity Index = Demand × (1 − Competition Penalty)
This allows structured evaluation of market attractiveness rather than simple POI counting.
________________________________________
3️⃣ Actor #7 Compatibility (Apify Integration)
Supports:
•	POI datasets with coordinates
•	Coverage-based datasets
•	Catchment analysis outputs
•	Competitor tagging via keywords
•	Automatic schema normalization
Can connect directly to:
•	Apify Run ID
•	Apify Dataset ID
________________________________________
4️⃣ Executive Memo Export
Generates structured PDF including:
•	KPI summary
•	Opportunity classification
•	Competitive interpretation
•	Economic definitions
•	Multi-radius comparison table
•	Top result breakdown
Designed for:
•	Internal strategy teams
•	Real estate committees
•	Retail investors
•	Franchise evaluation
________________________________________
🛠 Technology Stack
•	Streamlit (interactive UI)
•	Pandas / NumPy (analysis engine)
•	ReportLab (professional PDF export)
•	Apify API integration
•	Geospatial haversine calculations
________________________________________
📊 Example Use Cases
Retail Chain Expansion
Identify underserved micro-zones with low density and moderate competitor share.
Pharmacy Network Planning
Assess coverage adequacy within urban and suburban radii.
Restaurant Market Entry
Evaluate saturation vs demand proxy using reviews and rating signals.
Investment Committee Review
Export structured PDF for board-level discussion.
________________________________________
🧮 Analytical Framework
The model incorporates:
•	Density normalization
•	Competitor ratio weighting
•	Distance-based accessibility
•	Rating-based quality proxy
•	Log-scaled demand proxy (reviews)
The scoring system balances:
•	Demand
•	Accessibility
•	Quality
•	Competitive saturation
________________________________________
🔐 Secure Deployment
Environment variables supported:
APIFY_TOKEN=your_token_here
Secrets stored via:
.streamlit/secrets.toml
Sensitive credentials are not committed.
 
🚀 Local Installation
git clone https://github.com/YOUR_USERNAME/location-intelligence-dashboard.git
cd location-intelligence-dashboard

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
 
💼 Commercial Availability
This dashboard is available as:
•	Customizable analytics tool
•	White-label deployment
•	Catchment intelligence module
•	Retail expansion advisory support
•	Apify-integrated spatial engine
For custom deployments or consulting work, contact via Upwork

<img width="468" height="647" alt="image" src="https://github.com/user-attachments/assets/eed6663e-6c42-43cd-8ffd-bfe1a1877eb2" />
