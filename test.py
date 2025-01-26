import gradio as gr
from openai import OpenAI
import pdfplumber
import os
import tempfile
from dotenv import load_dotenv
import traceback
import pandas as pd

# Load environment variables and API keys
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)


def extract_text_and_tables_from_pdf(pdf_path):
    """
    Extract text and tables from a PDF using pdfplumber.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        str: Extracted text and tables in a formatted string
    """
    try:
        extracted_content = []

        with pdfplumber.open(pdf_path) as pdf:
            # Process each page
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text
                page_text = page.extract_text() or ""
                extracted_content.append(f"--- Page {page_num} ---\n{page_text}\n")

                # Extract tables
                tables = page.extract_tables()
                if tables:
                    for table_num, table in enumerate(tables, 1):
                        # Convert table to DataFrame for better formatting
                        df = pd.DataFrame(table[1:], columns=table[0])
                        table_str = f"\nTable {table_num} on Page {page_num}:\n"
                        table_str += df.to_string(index=False)
                        extracted_content.append(table_str + "\n")

        return "\n".join(extracted_content)

    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"


def summarize_soil_report(content):
    """
    Summarize the soil report using GPT-4.

    Args:
        content (str): Extracted text and tables from the PDF

    Returns:
        str: Detailed summary of the soil report
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert assistant in soil science. Analyze the soil report and provide a detailed summary for each sample.Make sure you analyse each page of the soil report. Only list the points that needs to be addressed,like lower or higher levels of chemicals . Present the information in a clear, organized manner."
                },
                {
                    "role": "user",
                    "content":f"Provide an analysis of this soil report,list out the important points that need to be addressed. Make sure to analyse of all pages of soil report pdf:\n\n{content}"
                }
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error summarizing soil report: {traceback.format_exc()}"


def answer_query(summary, query):
    """
    Answer specific questions about the soil report.

    Args:
        summary (str): Summary of the soil report
        query (str): User's specific question

    Returns:
        str: Detailed answer based on the soil report
    """
    if not query.strip():
        return "Please enter a question to get an answer."

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": "You are a soil science expert. Answer the user's question based on the provided soil report summary. Be specific and refer to the data in the summary when relevant. If the information is not in the summary, explain that clearly."
                },
                {
                    "role": "user",
                    "content": f"Using this soil report summary:\n{summary}\n\nAnswer this specific question: {query}"
                }
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error answering query: {traceback.format_exc()}"


def get_fertilizer_recommendations(summary):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert in soil science and fertilizers. Based on the soil analysis and the list of products provided, 
                    recommend specific fertilizer products.Make sure you dont suggest fertlisers that are rich in chemicals which already have higher levels in the soil report.Suggest Fertilisers for chemicals that have resulted in low levels in the soil report.
                    data-nitrophoska-extra,"Give your veges a nutrient spruce up with this tailored mix, developed for speciality crops in glasshouses or home gardens.    Great for specialty crops and all crops kept under glass or in the home garden.   Can also be used on lawns.   Contains potassium sulfate so it can be used on chloride-sensitive crops such as flowers, potatoes, lettuce, tomatoes, strawberries, citrus, and avocados. ",3.8,1.2,12,5.2,14,8(:The number in front of each product's description is the ratio of Calcium,Magnesium,Nitrogen,Phosphorus,Potassium,Sulphur respectively)
                    cropmaster®-dap,"Cost-effective option with a great balance of nitrogen and phosphate, for use on various crops, from field crops to green feed brassicas, and animal pastures.    Used on field crops, greenfeed brassicas, sheep, beef and dairy pasture.   Has good spreading and flow characteristics. ",0,0,17.6,20,0,1
                    potash-gold-14-7-14,Geared towards valuable crops that are sensitive to chloride and salt.The sulfate-sulfur is easily absorbed by your plants.   The macronutrient mix provided will suit most horticultural needs. ,2.4,0,14.3,7,14.5,6.7
                    cropmaster®-brassica-boron-blend,"Supplies the vital elements for brassica growth, nitrogen and phosphorus. Potassium is also added for areas with low to moderate soil potash levels.    Offers Nitrogen (13.9%), Phosphorus (15.4%), and Potassium (9.5%) with added Boron for forage brassicas. Nitrogen and Phosphorus are crucial for brassica growth.   Added Boron helps prevent disease (in boron-deficient crops). ",0,0,13.6,15.4,9.5,0.8
                    35-potash-gold-super-0-6-15,Geared towards valuable crops that are sensitive to chloride and salt.The sulphate-sulphur is easily absorbed by your plants.   The macronutrient mix provided will suit most horticultural needs. ,13,0,0,5.9,14.5,13.5
                    cropmaster®-brassica-mix,"Supplies the vital elements of nitrogen and phosphorus and potassium, for brassica to grow and thrive. Potassium is also added for areas with low to moderate soil potash levels.    Offers nitrogen (13.9%), phosphorus (15.4%), and potassium (9.5%) – all are crucial for brassica growth.   Potassium is also added for areas with low to moderate soil potash levels. ",0,0,14.1,16,10,0.8
                    nitro-s™,"  A well-balanced blend of N Protect (65%) and Sulphur 90 (35%).   Nitrogen provides protection from the volatilisation losses, and Sulphur 90 provides short and medium-term sulphur availability.",0,0,29.8,0,0,31.5
                    aglime,"Using lime boosts soil health and pasture growth by changing the soil pH, which helps plants absorb nutrients    Handy for managing soil acidity in New Zealand – without lime, soil pH naturally decreases over time.   Essential for cultivating legumes in NZ.   The lower the starting pH and the higher the lime application, the better the pasture growth. ",0,0,0,0,0,0
                    flexi-n-north-island,"coated urea product designed for mixing with superphosphate for spreading 
                     ",0,0.5,45.3,0,0,0
                    nitro-s™-1,  Contains Urea (65%) and Sulphur 90 (35%).   Fast acting nitrogen form of Urea with the Sulphur 90 providing short and medium-term sulphur availability.,0,0,29.9,0,0,31.5
                    cropmaster®-16-high-k-bulk,"A great balance of nitrogen (15.5%), phosphate (7%) and potassium (22.5%) for field crops, greenfeed brassicas, sheep, beef and dairy pasture.    A great blend of DAP with Urea and chipped potassium chloride.   Cost-effective option especially post supplement harvest (Hay/Silage).   Designed to spread evenly and smoothly, for a consistent coverage. ",0,0,15.4,7,22.5,0.4
                    potash-gold-15-10-10,Geared towards valuable crops that are sensitive to chloride and salt.    The sulfate-sulfur is easily absorbed by your plants.   The macronutrient mix provided will suit most horticultural needs. ,0,0,14.2,10,9.5,10.8
                    ureammopot,"  Ideal blend to kick-start spring growth, to provide immediately avail N, S and P, especially in soils that have lost winter nutrients due to high rainfall.   Great for the West Coast, Golden Bay and Taranaki areas.",0,0,25.7,0,10,9.7
                    dicalcic-high-s,"  Maintains the pH of your soil.   Provides a slower, more continuous release of phosphate over time as some phosphate reversion occurs in manufacture.  Reversion refers to the process where some of the soluble phosphate changes back into a less soluble form, making it available to plants more slowly over time.",26.9,0,0,4.2,0,9.3
                    ammo-phos®-map,"Great option for use with Hycrop pea fertiliser mixes for potatoes and forage brassica crops. It can also be used in base fertiliser mixes that are crafted for cereals and maize.    Good when a high phosphate supply is needed.   Well-suited for use with equipment that places fertiliser or seeds directly in the soil during planting, either in rows (drilling) or in narrow strips (banding).   Less risk of seed burning than DAP because it releases almost no free ammonia.   MAP-based products perform better on alkaline soils than DAP. ",0,0,10,22,0,1
                    potash-gold-7-15-13,Geared towards valuable crops that are sensitive to chloride and salt.    The sulfate-sulfur is easily absorbed by your plants.   The macronutrient mix provided will suit most horticultural needs. ,0,0,7,15.4,12.5,6.1
                    sechura-rpr, ,28,0,0,10.7,0,0
                    cobalt-super-1kg-cobalt-sulphate-is-21-co,"  Cobalt supplement for the prevention and treatment of cobalt deficiency   Cobalt deficiency is common in North Island pumice soils, parts of Southland and some areas in the top if the South Island.   The addition of cobalt will lift the pasture Cobalt levels passing to livestock through ingestion.",20,0,0,9,0,11
                    lime-reverted-super,"Lime Reverted Super is an economic option when inputs of P and S, as well as a boost to pH levels are required. The reaction of Lime and Superphosphate makes the resulting mix safe for mixing with seeds.
                     ",24,0,0,6.8,0,8.3
                    moly-sulphur-super-30-120g-mo,A blend of Sulphur Super 30 with trace amounts of Molybdenum. Molybdenum deficiency is more common in soils with lower   Provides plants with a readily available supply of molybdenum   Is particularly important for nitrogen fixation by leguminous plants   Minute quantities of product are required for optimum legume growth   Ideally applied as a maintenance programme,16,0,0,7,0,30.1
                    10-potash-super,"A readily available source of phosphorous, sulphur and potassium.
                      Usually applied as a base dressing or as a maintenance dressing.   An effective source of phosphorous, sulphur and potassium.   There&rsquo;s a range of differing ratios of superphosphate and potassium chloride in these various potash mixes, making it easy to select the right product for your soil, depending on your potassium availability.",18,0,0,8.1,5,9.9
                    cobalt-super-1-5kg-cobalt-sulphate-is-21-co,"Cobalt Super is a blend of Superphosphate with Cobalt. Cobalt is essential to address animal health issues in cobalt deficient soils. 
                      Cobalt supplement for the prevention and treatment of cobalt deficiency   Cobalt deficiency is common in North Island pumice soils, parts of Southland and some areas in the top if the South Island.   The addition of cobalt will lift the pasture Cobalt levels passing to livestock through ingestion.",20,0,0,9,0,11
                    moly-super-200g-mo,  Provides plants with a readily available supply of molybdenum   Is particularly important for nitrogen fixation by leguminous plants   Minute quantities of product are required for optimum legume growth   Ideally applied as a capital rate,20,0,0,9,0,11
                    selenium-super,"  Treatment and prevention of selenium deficiency in grazing animals   Selenium deficiency can cause white muscle disease, poor growth in calves, lambs and fawns.   Selenium deficiency can cause low fertility in sheep and poor milk production in cattle.",20,0,0,9,0,11
                    moly-super-120g-mo,Molybdenum deficiency is more common in soils with lower  Provides plants with a readily available supply of molybdenum. Is particularly important for nitrogen fixation by leguminous plants. Minute quantities of product are required for optimum legume growth. Ideally applied as a maintenance programme. ,20,0,0,9,0,11
                    dap-13-s,"Ideal pastoral option where nitrogen and phosphate are required, with both readily available and slower release sulphur.    Provides nitrogen (10.8%), phosphate (14.8%) & sulphur (12.6%) for existing pastures.   Advise prompt application to avoid product from becoming lumpy. ",6.4,0,10.6,14.8,0,12.6
                    20-potash-super-mag-n,"More powerful option for maintenance or pasture development, especially when there’s risk of low nitrogen and a fast-acting product is needed.    Get quicker results on forage crops or gardens, where plants need an immediate boost.   The nutrients are easily soluble, so plants can respond right away.   Contains more potassium, ideal where soils are potassium deficient. ",10.2,3.7,5.5,4.6,10,5.8
                    nitrogen-super," for use on pastures as well as sowing with cereals, where a nitrogen boost ",14,0,6,6.3,0,14.6
                    15-potash-serpentine-super,"A solid all-around option for planting new pastures, Serpentine Super is near pH neutral so there’s little risk of seed burn. •	A well-balanced mix of phosphorus, potassium, sulphate sulphur and magnesium.",12.8,4.7,0,5.7,7.5,7.3
                    20-potash-serpentine-super,"A reliable choice when developing new pastures, Serpentine Super is near pH neutral so there’s little risk of seed burn.    A great mix of phosphorus, potassium, sulphate sulphur and magnesium.   Contains more potassium, ideal where soils are potassium deficient. ",12,4.4,0,5.4,10,6.9
                    magnesium-super,A cost-effective way to provide magnesium to pastures where the soil is deficient.    Useful on yellow brown pumice soils where your soil type contains little or no magnesium. ,18.4,3.2,0,8.3,0,10.1
                    magnesium-oxide,"A great source of magnesium, suited to pastoral, arable and horticultural applications.    Cost-effective source of magnesium where large amounts are required.   Maintain or increase soil pasture magnesium levels, both for plant and animal requirements. ",0,40,0,0,0,0
                    sulphur-super-20,"A blend of Sulphur super 30 and Superphosphate, great for maintenance when topdressing isn’t done frequently.    This blend (50/50 Sulphate: elemental S), offers both fast-acting sulphate-S and medium-term release Elemental-S.   Different ratios will result in varying sulphur levels and distinct products, offering you more flexibility. ",18,0,0,8,0,20.6
                    15-potash-super-mag-n,"Ideal for everyday maintenance or pasture development, especially when there’s risk of low nitrogen and a fast-acting product is needed.    Perfect for quick results in forage crops, or gardens where plants need an immediate boost.   The nutrients are easily soluble, so plants can respond right away. ",10.8,4,5.9,4.8,7.5,6.2
                    sulphur-90,"Supplementing your soluble sulphate fertilisers with Sulphur 90 will keep sulphur levels up throughout the growing season, so plants get a steady supply.    It’s easier for plants to absorb (as the elemental sulphur particles convert to sulphate sulphur).   Supplementing soluble sulphate fertilisers with Sulphur 90 will ensure sulphur isn’t depleted in the growing season.   Sulphur 90 is a fertiliser made up of thousands of tiny particles of elemental sulphur per split pea shaped pastille. ",0,0,0,0,0,90
                    esta®-kieserite-granular,"Popular for use on fast-growing crops, gardens and maize, containing magnesium and sulphur in forms that plants can easily absorb.    A great source of magnesium and sulphur source for all short-term crops and in all types of soil, regardless of soil pH.   Contains magnesium and sulphur in water soluble forms, which the plants can absorb. ",0,15,0,0,0,20
                    sulphur-super-15,"Great for maintenance when topdressing is less frequent, and ideal when there's a higher demand for sulphur compared to phosphorus. Suitable for higher rainfall areas or where biennial application is needed.    Quick acting sulphate-S and medium-term release Elemental-S.   Flexibility, as different ratios produce different sulphur levels. ",19.2,0,0,8.6,0,14.8
                    20-potash-sulphur-super,"A fast-acting choice for routine maintenance on potassium deficient soils.    A great mix of soluble phosphate, potassium, calcium and sulphur.   Has both quick-acting sulphate sulphur, and medium-term release elemental sulphur.   Suitable for maintenance on potassium deficient soil, or post-harvest of supplements (Hay/Silage). ",14.4,0,0,6.4,10,16.4
                    super-mag-n,"Developed for pastoral maintenance or development  A multi-nutrient fertiliser which suits many different land uses, especially when five of the major nutrients are required.   Ideal for forage cropping, arable or horticultural situations where fast responses are needed, as nutrients are quickly absorbed by plants.",12.8,4.7,6.9,5.7,0,7.3
                    50-potash-super,"A readily available source of phosphorous, sulphur and potassium  Usually applied as a base dressing or as a maintenance dressing. An effective source of phosphorous, sulphur and potassium. There&rsquo;s a range of differing ratios of superphosphate and potassium chloride in these various potash mixes, making it easy to select the right product for your soil, depending on your potassium availability. ",10,0,0,4.5,25,5.5
                    sulphur-super-30,"Excellent option for maintenance when topdressings are infrequent, containing fast-acting Sulphate-S and medium-term release Elemental-S.    Fast acting, as granules oxidise in soil over about 12-24 months. ",16,0,0,7,0,30.1
                    30-potash-serpentine-super," for planting new pastures  A well-balanced mix of phosphorus, potassium, sulphate and magnesium. <br />A good general fertiliser that can be used when sowing new pastures ",10.5,3.9,0,4.7,15,6
                    20-potash-super,"A readily available source of phosphorous, sulphur and potassium.
                      Usually applied as a base dressing or as a maintenance dressing.   An effective source of phosphorous, sulphur and potassium.   There&rsquo;s a range of differing ratios of superphosphate and potassium chloride in these various potash mixes, making it easy to select the right product for your soil, depending on your potassium availability.",16,0,0,7.2,10,8.8
                    40-potash-super,"A readily available source of phosphorous, sulphur and potassium.  Usually applied as a base dressing or as a maintenance dressing.   An effective source of phosphorous, sulphur and potassium.   There&rsquo;s a range of differing ratios of superphosphate and potassium chloride in these various potash mixes, making it easy to select the right product for your soil, depending on your potassium availability.",12,0,0,5.4,20,6.6
                    30-potash-sulphur-super,"  Contains soluble phosphate, potassium, calcium, and sulphur Has both quick acting sulphate S &amp; medium term release elemental S Suitable type of fertiliser for mostly maintenance applications on pastoral soils ",12.6,0,0,5.6,15,14.4
                    pasture-6-ravensdown-bulk,"  Used for pastoral maintenance or development, particularly where nitrogen status of the soil is expected to be low.   Ideal for forage cropping, to arable or horticultural situations.   Use when immediate absorption and responses are needed.",12.1,0,5.5,5.5,6,13
                    lucerne-mix-te,"  Contains the essential trace elements for healthy lucerne production.   Use only once every 2-3 years on lucerne, to ensure there is no oversupply of key trace elements.",12.4,0,0,5.5,14.7,13.2
                    cropmaster®-11,"  Provides a balanced mix of N (10.8%), P (12%) and K (20%).   Ideal on field crops and greenfeed brassicas.",0,0,10.6,12,20,0.6
                    cropmaster®-13,"  Provides a balanced mix of N(12.6%), P (14%) and K (15%) specifically for greenfeed brassicas, particularly in areas of higher rainfall or where potassium is limited.   Potassium plays an important function in plant water regulation.",0,0,12.3,14,15,0.7
                    compound-premium-sop,"Ideal for avocados, grapes, lettuces, melons, cucumbers, peppers, and all crops under glass and home gardens.
                      Contains N in both the nitrate and ammoniacal form for immediate and rapid availability to the germinating seedling or transplant.   Contains potassium in the form of SOP (sulphate of potash) so it can be used on chloride-sensitive crops.   Has a near-neutral effect on the soil pH, which is important when banding the product&#8203;.   Ideal product for row crop application and precision planting.   An all-round option for any avid home gardener.",0,1.2,12,5.2,14.1,10
                    cropmaster®-15," 15 was originally designed for cereal cropping & has a range of other uses including maize, grass re-establishment, and pasture for hay
                      A cost-effective product providing (15.1%), P (10%), K (10%) and S (7.7%).   Ideal for field crops, sheep, beef and dairy pasture.   Great for cropping situations where potassium is required.",0,0,14.8,10,10,7.4
                    cropmaster®-20,"An affordable fertiliser that offers easily accessible nitrogen in the ammonium form. It also contains phosphate and sulphate sulphur.    Delivers nitrogen (18.8%), phosphorus (10%), and sulphur (12%) for various crops like field crops, green feed brassicas, and pastures for sheep, beef, and dairy.   Ideal for cropping situations where potassium isn't needed. ",0,0,18.8,10,0,12
                    15-potash-super,"Potash Super is a blend of Superphosphate and Potassium Chloride that helps support plant growth. It’s usually applied to land before sowing with seed or to established pastures for maintaining.    Is a good source of phosphorous, sulphur, and potassium to support plant growth.   Select the right product for your situation from the wide range of Potash Supers.   Helps with maintaining established pastures.   Great to use when preparing soil for planting. ",17,0,0,7.7,7.5,9.4
                    15-potash-sulphur-super,"A blend of Superphosphate, Sulphur Super 30 and Potassium Chloride that helps support plant growth. Best suited for use on established pastures for maintaining the soil.    Contains soluble phosphate, potassium, calcium, and sulfur to support plant growth. ",15.3,0,0,6.8,7.5,17.5
                    30-potash-super,"Potash Super is a blend of Superphosphate and Potassium Chloride that helps support plant growth. It’s usually applied to land before sowing with seed or to established pastures for maintaining.    Is a good source of phosphorus, sulfur, and potassium to support plant growth.   Select the right product for your situation from the wide range of Potash Supers.   Helps with maintaining established pastures.   Great to use when preparing soil for planting. ",14,0,0,6.3,15,7.7
                    ravensdown-dicalcic-phosphate®,"Ravensdown Dicalcic Phosphate is a manufactured fertiliser, containing plant-available phosphorus, sulphur and agricultural Ag Lime for optimum pasture growth. It is only available in the North Island.    Extends pasture growth.   Ideal for regular maintenance or developing new land. ",28.6,0,0,4.1,0,5.1
                    potassium-chloride-granular-1,"A high content potassium fertiliser that readily dissolves for ease of use.    Ideal for Pastoral uses, not recommended on chloride sensitive crops.   Mixes well with all other fertilisers for targeted applications.   Should be used in potassium deficient soils and post-harvest of supplements (Hay/Silage). ",0,0,0,0,50,0
                    sulphate-of-potash-granular-1,"Used in fertilisers, Sulphate of Potash – Granular supports plant growth, disease resistance and water preservation. The essential sulphur content is derived naturally from plants.    It’s virtually chloride free and has a low salt index making it ideal for use on crops that are sensitive to these two compounds.   The low salt index supports water preservation by plants.   Mixes well with all other fertilisers for ease of targeted application. ",0,0,0,0,41.5,18
                    serpentine-super-drilling-super,"Serpentine, also known as drilling Super, is a reliable all-around fertiliser. It gives your crops an instant boost of phosphate, sulphur, and will not cause germination injury to seed when applied at recommended rates.    Designed for pastoral agriculture, with better handling and mixing properties.   Used in forage cropping and vegetable production.   Safe for germinating seeds when applied at recommended rates. ",15,5.5,0,6.7,0,8.6
                    flexi-n-south-island-1,"Tailored for South Island weather and soil conditions, Flexi-N can be mixed with Superphosphate. Prompt spread is recommended.     One application does the trick.   Ability to apply N, P, K, and S.   Suitable for aerial or ground spreading. ",0,2.4,43.2,0,0,0
                    ammo-36-pro-1,"Great for areas with low sulphate and lots of rain, and boosts nitrogen and sulphate levels ready for spring. Ammo 36 Pro contains N-Protect that helps urea work better, saving nitrogen loss.    Great for use in late winter/early spring, to enhance dairy pasture growth.   Keeps nitrogen where plants need it most – the roots.   Research indicates it can reduce nitrogen loss (through volatilisation) by 50%.   Less ammonia gas lost, so is better for the environment. ",0,0,35.5,0,0,9.2
                    calcium-ammonium-nitrate-can-1,"Nearly pH-neutral, ideal for perennial fruit crops where mixing lime into the soil is tricky. Combines fast-acting nitrate-nitrogen with longer-lasting ammonium-nitrogen.     Can be used on soils that have a low pH (without lowering further).   Suitable to use in all seasons.   Includes ammonium nitrate, classified as safe and non-flammable. ",5.3,2.4,27,0,0,0
                    moly-sulphur-super-30-200g-mo-1,"Sulphur Super 30 contains both sulphate and elemental sulphur, providing immediate available sulphur as well as slower availability. Used in a maintenance role where topdressings are infrequent.     Has both quick acting Sulphate-S & medium-term release Elemental-S.   Used in a capital role where the requirement for S is much higher than for P.   The particles of elemental sulphur within the granule will oxidise in soil over approximately 12-24 months.   Blend contains maintenance rates of molybdenum. ",16,0,0,7,0,30.1
                    ammo-31-pro-1,"Perfect for low-sulphate, high-rainfall areas, boosting nitrogen and sulphate to get ready for spring growth. Ammo 31 Pro contains N-Protect that helps urea work better, saving nitrogen loss.    Great for use in late winter/early spring, to enhance pasture growth.   Keeps nitrogen where plants need it most – the roots.   Research indicates it can reduce nitrogen loss (through volatilisation) by 50%.   Less ammonia gas lost, so it is better for the environment. ",0,0,30.3,0,0,13.8
                    triple-super-2,"Triple Super is a Superphosphate product that doesn’t contain sulphur. It is a manufactured fertiliser, containing plant-available phosphorus for optimum pasture growth.    Suitable for crops that need phosphate but not sulphur.   It's high solubility makes it easier for spreading.   The high phosphate concentration means that it’s cost competitive when it comes to spreading and easier to transport. ",16,0,0,20.5,0,1
                    ammo-36™-1,"Crafted for farms for use in late winter and early spring, when sulphate is low and rainfall is high.    Includes the essential mix of nitrogen and sulphate, ready for spring.   Blend of ammonium sulphate and Urea at a ratio of 40:60. ",0,0,35.6,0,0,9.2
                    ammo-31™-1,"Perfect for boosting pasture in late winter and early spring, with low soil sulphate and high rainfall.     Enhances nitrogen and sulphate levels ready for spring growth.   Perfect blend of ammonium sulphate and Urea at a ratio of 60:40. ",0,0,30.4,0,0,13.8
                    urea-1,"Cost effective nitrogen source. Nitrogen is the most important nutrient for plant growth. Suitable for all agriculture, horticulture and forestry farming systems.    Highest quick release nitrogen (N) content of any solid fertiliser in NZ.   Increase your dry matter production to meet animal feed needs, like hay or silage. ",0,0,46,0,0,0
                    granular-ammonium-sulphate-1,"A well-balanced mix of nitrogen and sulphur your plants will love, and handy in the short term when soil needs extra sulphur.    Gives plants and soil the perfect balance of nitrogen and sulphur they need to thrive.   Helps fix short-term sulphur shortages in both crops and pastures.   The safest nitrogen product to mix with fertilisers that don't have superphosphate in them. ",0,0,20,0,0,23
                    n-protect®,"N-Protect is designed to slow down the release of ammonia, keeping more nitrogen for your plants rather than losing it into the air. By doing so, you're not only improving your crops but also reducing your environmental footprint—a win-win.    Coated urea reduces nitrogen losses, saving money in the long run.   Your plants can make the most of extra nitrogen in their roots.   Research shows that using the urease inhibitor coating can cut nitrogen loss by 50%, making your farm more eco-friendly. ",0,0,45.9,0,0,0
                    superphosphate-1,Superphosphate is a manufactured fertiliser containing plant-available phosphorus and sulphur for optimum pasture growth    Cost-effective fertiliser to extend pasture growth.   Perfect for regular maintenance or developing new land. ,20,0,0,9,0,11
                    avocado-regular-mix-te-1,"Designed to provide all the nutritional requirements for an avocado tree in one mix. The potassium in the mix is exclusively Potassium Sulphate, making it safe for avocado trees, which are sensitive to chloride.
                      Provides all the nutrients (NPKS + TE) an avocado tree requires in one mix. Contains only Potassium Sulphate so it&rsquo;s great for use on chloride sensitive crops. Contains enough Boron and Zinc to meet plant demands. ",3,2.7,9.6,4.2,18.7,9.9
                    dolomite,"Ideal in dairy and horticultural systems. Dolomite is a naturally occurring source of both magnesium and calcium carbonate.
                        Combined source of magnesium and lime   Magnesium content in a readily available form due to the particle size of the product ",23,11,0,0,0,0
                    lawn-fertiliser-1,"Crafted to give you lush green grass by balancing nitrogen and phosphorus, plus a touch of iron sulphate to keep moss at bay.
                      Nitrogen applied in the form of Ammonium Sulphate helps to lower the soil pH, discouraging earthworms. This product can even add vibrancy to grass colour. ",5,0,14.5,2.3,0,19.7
                    garden-fertiliser-1,"Adding this balance blend of garden fertiliser offers your soil a healthy boost of essential nutrients.
                      When your soil is in good shape and has the right nutrients, your plants can grow, produce, and flourish the way they should. A balanced blend of Superphosphate, Potassium Sulphate, Ammonium Sulphate, and Magnesium Oxide with the addition of a small amount of Sodium Molybdate and Sodium Borate. ",9.4,1,6.6,4.3,7.1,15.8"""
                },
                {
                    "role": "user",
                    "content": f"Based on this soil analysis summary and the list of products that you have, provide detailed fertilizer recommendations:\n\n{summary}"
                }
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating recommendations: {traceback.format_exc()}"


class State:
    def __init__(self):
        self.summary = None


state = State()


def process_pdf(pdf_file):
    """
    Process the uploaded PDF file.

    Args:
        pdf_file (bytes): Uploaded PDF file

    Returns:
        Tuple of outputs for Gradio interface
    """
    if pdf_file is None:
        return "Please upload a PDF file.", gr.update(visible=False), gr.update(visible=False), gr.update(
            visible=False), None, None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_file)
            temp_pdf_path = temp_pdf.name

        content = extract_text_and_tables_from_pdf(temp_pdf_path)
        if content.startswith("Error"):
            return content, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), None, None

        summary = summarize_soil_report(content)
        state.summary = summary

        try:
            os.unlink(temp_pdf_path)
        except:
            pass

        # Make query box, buttons, and output boxes visible
        return (summary,
                gr.update(visible=True),  # Query box
                gr.update(visible=True),  # Ask Question button
                gr.update(visible=True),  # Get Recommendations button
                None,  # Clear query output
                None)  # Clear recommendations output

    except Exception as e:
        error_msg = f"Error in processing: {traceback.format_exc()}"
        return error_msg, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), None, None


def process_query(query):
    """
    Process user's query about the soil report.

    Args:
        query (str): User's specific question

    Returns:
        str: Answer to the query
    """
    if state.summary is None:
        return "Please upload a soil report first."
    return answer_query(state.summary, query)


def process_recommendations():
    """
    Generate fertilizer recommendations.

    Returns:
        str: Fertilizer recommendations
    """
    if state.summary is None:
        return "Please upload a soil report first."
    return get_fertilizer_recommendations(state.summary)


# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as gui:
    gr.Markdown("# Soil Report Analyzer")
    gr.Markdown(
        "Upload a soil report PDF to get an instant comprehensive analysis, then ask questions or get fertilizer recommendations.")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload Soil Report PDF",
                type="binary",
                file_types=[".pdf"]
            )

            # Query section
            with gr.Group(visible=False) as query_group:
                query_input = gr.Textbox(
                    label="Ask a question about the soil report",
                    placeholder="Example: What is the pH level in sample 1?",
                )
                ask_button = gr.Button("Ask Question", variant="primary")

            # Recommendations button
            get_recommendations_button = gr.Button(
                "Get Fertilizer Recommendations",
                visible=False,
                variant="secondary"
            )

        with gr.Column(scale=2):
            # Output boxes
            summary_output = gr.Textbox(
                label="Soil Report Analysis",
                lines=12,
                placeholder="Your soil report analysis will appear here..."
            )

            query_output = gr.Textbox(
                label="Answer",
                lines=4,
                placeholder="Your answer will appear here..."
            )

            recommendations_output = gr.Textbox(
                label="Fertilizer Recommendations",
                lines=8,
                placeholder="Fertilizer recommendations will appear here..."
            )

    # Event handlers
    file_input.upload(
        process_pdf,
        inputs=[file_input],
        outputs=[
            summary_output,
            query_group,
            ask_button,
            get_recommendations_button,
            query_output,
            recommendations_output
        ]
    )

    # Handle query submission
    ask_button.click(
        process_query,
        inputs=[query_input],
        outputs=[query_output]
    )

    # Also allow Enter key to submit query
    query_input.submit(
        process_query,
        inputs=[query_input],
        outputs=[query_output]
    )

    # Handle recommendations request
    get_recommendations_button.click(
        process_recommendations,
        inputs=[],
        outputs=[recommendations_output]
    )

if __name__ == "__main__":
    gui.launch(share=True)