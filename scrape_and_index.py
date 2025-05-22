import requests
from bs4 import BeautifulSoup
import faiss
import numpy as np
import pickle
from fastembed import TextEmbedding

print("Initializing fastembed model...")
embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
print("Fastembed model loaded!")

def scrape_magicbricks_jaipur(max_pages=5):
    base_url = "https://www.magicbricks.com/property-for-sale/residential-real-estate?bedroom=2,3&proptype=Multistorey-Apartment,Builder-Floor-Apartment,Penthouse,Studio-Apartment,Residential-House,Villa&cityName=Jaipur"
    listings = []
    for page in range(1, max_pages+1):
        url = f"{base_url}&page={page}"
        response = requests.get(url, headers={"User-Agent":"Mozilla/5.0"})
        print(f"Magicbricks HTTP status (page {page}):", response.status_code)
        soup = BeautifulSoup(response.text, "html.parser")
        cards = soup.select('.mb-srp__card')
        print(f"Found {len(cards)} cards on Magicbricks page {page}.")
        if not cards:
            print("No cards found! Ending pagination.")
            break
        for card in cards:
            title = card.select_one('.mb-srp__card--title')
            price = card.select_one('.mb-srp__card__price--amount')
            location = card.select_one('.mb-srp__card__address')
            if location is None:
                location_alt = card.select_one('.mb-srp__card__location')
                location_text = location_alt.text.strip() if location_alt else ""
            else:
                location_text = location.text.strip()
            # Area
            area = card.select_one('.mb-srp__card__summary--value')
            area_text = area.text.strip() if area else ""
            # Bathrooms and property type from summary
            summary_items = card.select('.mb-srp__card__summary--item')
            bathrooms = ""
            prop_type = ""
            for item in summary_items:
                label = item.select_one('.mb-srp__card__summary--label')
                val = item.select_one('.mb-srp__card__summary--value')
                if label and val:
                    label_text = label.text.strip().lower()
                    val_text = val.text.strip()
                    if "bath" in label_text:
                        bathrooms = val_text
                    if "property type" in label_text or "type" in label_text:
                        prop_type = val_text
            posted = card.select_one('.mb-srp__card__date')
            posted_date = posted.text.strip() if posted else ""
            # Property link
            link_tag = card.select_one('a.mb-srp__card--title')
            link = ""
            if link_tag and link_tag.has_attr('href'):
                href = link_tag['href']
                if not href.startswith('http'):
                    link = "https://www.magicbricks.com" + href
                else:
                    link = href
            listings.append({
                "title": title.text.strip() if title else "",
                "location": location_text,
                "price": price.text.strip() if price else "",
                "area": area_text,
                "bathrooms": bathrooms,
                "property_type": prop_type,
                "posted_date": posted_date,
                "link": link,
                "source": "Magicbricks"
            })
    return listings

def scrape_housing_jaipur():
    url = "https://housing.com/buy-flats-in-jaipur-for-sale-srpid-P268dw8h2rtw7ykpy"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    }
    response = requests.get(url, headers=headers)
    print("housing.com HTTP status:", response.status_code)
    soup = BeautifulSoup(response.text, "html.parser")
    listings = []
    cards = soup.select('article[data-testid="srp-card"]')
    print(f"Found {len(cards)} cards on housing.com page.")
    if not cards:
        print("No cards found! Printing a snippet of the HTML for debugging:")
        print(response.text[:2000])
        return []
    for card in cards:
        title = card.select_one('h2[data-testid="listing-title"]')
        price = card.select_one('span[data-testid="listing-price"]')
        location = card.select_one('div[data-testid="listing-location"]')
        # Area
        area = card.select_one('div[data-testid="listing-area"]')
        area_text = area.text.strip() if area else ""
        # Bathrooms and property type
        bathrooms = ""
        bath_icon = card.find('svg', attrs={"data-testid": "bathroom-icon"})
        if bath_icon:
            bath_parent = bath_icon.find_parent('li')
            if bath_parent:
                bathrooms = bath_parent.text.strip()
        prop_type = ""
        type_field = card.select_one('div[data-testid="listing-type"]')
        if type_field:
            prop_type = type_field.text.strip()
        posted_date = ""
        posted = card.select_one('div[data-testid="listing-posted-date"]')
        if posted:
            posted_date = posted.text.strip()
        # Property link
        link_tag = card.select_one('a')
        link = ""
        if link_tag and link_tag.has_attr('href'):
            href = link_tag['href']
            if not href.startswith("http"):
                link = "https://housing.com" + href
            else:
                link = href
        listings.append({
            "title": title.text.strip() if title else "",
            "location": location.text.strip() if location else "",
            "price": price.text.strip() if price else "",
            "area": area_text,
            "bathrooms": bathrooms,
            "property_type": prop_type,
            "posted_date": posted_date,
            "link": link,
            "source": "housing.com"
        })
    return listings

def get_fastembed_embeddings(texts):
    print("Getting fastembed embeddings (this may take several seconds)...")
    embeddings = list(embed_model.embed(texts))
    embeddings = np.vstack(embeddings).astype("float32")
    print("Embeddings shape:", embeddings.shape)
    return embeddings

def main():
    print("Scraping Magicbricks...")
    magicbricks = scrape_magicbricks_jaipur()
    print(f"Magicbricks: {len(magicbricks)} listings scraped.")
    print("Sample:", magicbricks[:2])

    print("Scraping housing.com...")
    housing = scrape_housing_jaipur()
    print(f"housing.com: {len(housing)} listings scraped.")
    print("Sample:", housing[:2])

    properties = magicbricks + housing
    print(f"Total properties scraped: {len(properties)}")
    if not properties:
        print("ERROR: No properties scraped! Check your selectors or your network.")
        return

    # Use more fields for embedding (for richer search)
    texts = [
        f"{p['title']} {p['location']} {p['area']} {p['price']} {p['bathrooms']} {p['property_type']} {p['posted_date']}"
        for p in properties
    ]
    embeddings = get_fastembed_embeddings(texts)

    print("Building FAISS index and saving files...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, "faiss_index.bin")
    with open("metadata.pkl", "wb") as f:
        pickle.dump(properties, f)
    print("Done! Saved faiss_index.bin and metadata.pkl")

if __name__ == "__main__":
    main()