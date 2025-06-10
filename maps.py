MAIL_DELIVERY_MAPS = {
    "default": {
        # ── overall dimensions ──────────────────────────────────────────────
        "size": (10.0, 10.0),          # W, H  in metres

        # ── fixed locations ────────────────────────────────────────────────
        "depot": (1.0, 1.0),           # lone pick-up point

        # ── obstacles (axis-aligned boxes) ─────────────────────────────────
        #    xmin, ymin, xmax, ymax
        "obstacles": [
            (1, 3, 3, 8),      
            (3, 4, 7, 5),     
        ],
    }
}