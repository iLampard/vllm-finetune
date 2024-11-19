from easyllm_kit.utils import PromptTemplate


class SFTTemplate(PromptTemplate):
    """Custom prompt template for generating styling query."""

    @classmethod
    def create_default(cls) -> "SFTTemplate":
        template = '''### Search Intent Recognition
            Input: Query: {{ query }} + Image (optional) <image>  (Query priority)
            
            ### Intents & Attributes:
            
            1. **Styling** ("how to style", "outfit ideas")
               - 8 items: tops, bottoms, shoes, accessories
               - Extract: gender, season, occasion, style
            
            2. **Similar** ("find similar", "looking for like")
               - 8 matching items from the SAME CATEGORY as query product  
               - Extract: style, color, material, function
            
            3. **General** (all other queries)
               - 8 category items
               - Categories: Fashion, Beauty, Home, Tech, etc.
            
            ### Output Format
            Note: Use 0 for unspecified price range.
            Format the response as a valid JSON object.
            
            query: "Summer date outfits for women"
            
            ```json{
                "intention": "styling",
                "gender": "women",
                "price_range": {
                    "min": 0,
                    "max": 0
                },
                "attributes": [
                    "summer",
                    "dating",
                    "causal"
                ],
                "items": [
                       "chiffon floral dress for women",
                       "designer unique top for women", 
                       "high waist wide leg pants for women",
                       "fairy style sandals for women", 
                       "chain mini bag for women", 
                       "pearl earrings for women", 
                       "dating lipstick for women",
                       "stylish sunhat for women"
                ],
            }
            ```
            
    '''

        return cls(
            template=template,
            input_variables=['query']
        )