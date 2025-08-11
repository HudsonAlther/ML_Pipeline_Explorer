"""
Dataset configurations for ML Pipeline Explorer.
"""

DATASETS = {
    "stocks": {
        "path": "datasets/all_stocks_5yr.csv",
        "target": "AboveAvg",
        "prepare_fn": "prepare_stocks",
        "description": "Predict whether closing price is above its series mean",
        "business_value": "Directional price heuristic for retail traders",
        "prediction_context": {
            "what_is_predicted": "Whether Apple (AAPL) stock price will be above or below its average price",
            "prediction_values": {
                "0": "Below Average - Stock price is expected to be lower than its historical average",
                "1": "Above Average - Stock price is expected to be higher than its historical average"
            },
            "business_implication": "Helps traders decide whether to buy (above average) or sell (below average) based on current market conditions",
            "real_world_use": "Used by retail investors and day traders to make quick buy/sell decisions",
            "risk_note": "This is a simple heuristic and should not be the only factor in investment decisions"
        },
        "features_explanation": {
            "open": "Opening price of the stock for the day",
            "high": "Highest price reached during the trading day", 
            "low": "Lowest price reached during the trading day",
            "volume": "Number of shares traded that day"
        }
    },
    "terrorism": {
        "path": "datasets/globalterrorismdb_0718dist.csv",
        "target": "success",
        "prepare_fn": "prepare_terrorism",
        "description": "Predict whether reported attacks were successful",
        "business_value": "Triage and risk scoring for incident records",
        "prediction_context": {
            "what_is_predicted": "Whether a terrorist attack was successful or not",
            "prediction_values": {
                "0": "Unsuccessful - The attack did not achieve its intended goals",
                "1": "Successful - The attack achieved its intended goals"
            },
            "business_implication": "Helps security analysts prioritize threat assessments and allocate resources more effectively",
            "real_world_use": "Used by security agencies and risk assessment teams to evaluate threat levels and plan countermeasures",
            "risk_note": "This analysis is for security planning purposes and should be used responsibly"
        },
        "features_explanation": {
            "nkill": "Number of people killed in the attack",
            "nwound": "Number of people wounded in the attack"
        }
    },
    "netflix": {
        "path": "datasets/netflix_titles.csv",
        "target": "is_movie",
        "description": "Content classification for streaming platforms",
        "business_value": "Catalog organization and recommendation enhancement",
        "prediction_context": {
            "what_is_predicted": "Whether a Netflix title is a Movie (0) or TV Show (1)",
            "prediction_values": {
                0: "Movie",
                1: "TV Show"
            },
            "business_implication": "Improves content organization and recommendation algorithms",
            "real_world_use": "Streaming platforms use this to categorize content and enhance user experience",
            "risk_note": "For content organization purposes only"
        },
        "features_explanation": {
            "title": "The name of the content",
            "director": "Who directed the content",
            "cast": "Main actors in the content",
            "country": "Where the content was produced",
            "date_added": "When it was added to Netflix",
            "release_year": "When the content was originally released",
            "rating": "Content rating (PG, R, etc.)",
            "duration": "Length of the content",
            "listed_in": "Categories the content belongs to",
            "description": "Brief description of the content"
        }
    },
}
