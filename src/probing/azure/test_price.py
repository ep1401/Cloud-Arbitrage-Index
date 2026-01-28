from meta_cai_scheduler_azure import spot_price_latest, spot_price_delta

for region in ["eastus", "westus2"]:
    for sku in ["Standard_D2ads_v5", "Standard_D2s_v3", "Standard_E2s_v3"]:
        p, src = spot_price_latest(region, sku)
        d = spot_price_delta(region, sku)
        print(region, sku, "price", p, "src", src, "delta6h", d)
