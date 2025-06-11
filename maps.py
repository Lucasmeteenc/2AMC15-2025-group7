MAIL_DELIVERY_MAPS = {
    "default": {
        # ── overall dimensions ──────────────────────────────────────────────
        "size": (10.0, 10.0),          # W, H  in metres

        # ── fixed locations ────────────────────────────────────────────────
        "depot": (3.0, 1.0),           # lone pick-up point
        "delivery": (7.0, 9.0),      # lone drop-off point

        # ── obstacles (axis-aligned boxes) ─────────────────────────────────
        #    xmin, ymin, xmax, ymax
        "obstacles": [
            (1, 3, 3, 8),      
            (3, 4, 7, 5),     
        ],
    }
}

COMPLEX_DELIVERY_MAPS = {
    "default": {
        # ── overall dimensions ──────────────────────────────────────────────
        "size": (10.0, 10.0),          # W, H  in metres

        # ── fixed locations ────────────────────────────────────────────────
        "depot": (1.0, 1.0),           # lone pick-up point
        "chargers": [                  # three charger pads
            (7.0, 1.0),
            (6.0, 9.0),
        ],

        # ── obstacles (axis-aligned boxes) ─────────────────────────────────
        #    xmin, ymin, xmax, ymax
        "obstacles": [
            (1, 3, 3, 8),      
            (3, 4, 7, 5),     
        ],
    }
}

