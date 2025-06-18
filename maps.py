MAIL_DELIVERY_MAPS = {
    "default": {
        # ── overall dimensions ──────────────────────────────────────────────
        "size": (10.0, 10.0),          # W, H  in metres

        # ── fixed locations ────────────────────────────────────────────────
        "depot": (3.0, 1.0),           # lone pick-up point
        "delivery": (7.0, 9.0),      # lone drop-off point
        # "starting_position": (4.0, 6.0), # starting position of the robot

        # ── obstacles (axis-aligned boxes) ─────────────────────────────────
        #    xmin, ymin, xmax, ymax
        "obstacles": [
            (1, 3, 3, 8),      
            (3, 4, 7, 5),     
        ],
    },

    "empty": {
        # ── overall dimensions ──────────────────────────────────────────────
        "size": (10.0, 10.0),          # W, H  in metres

        # ── fixed locations ────────────────────────────────────────────────
        "depot": (1.0, 1.0),           # lone pick-up point
        "delivery": (8.0, 9.0),      # lone drop-off point
        # "starting_position": (5.0, 5.0), # starting position of the robot

        # ── obstacles (axis-aligned boxes) ─────────────────────────────────
        #    xmin, ymin, xmax, ymax
        "obstacles": [
            # (1, 3, 3, 8),      
            # (3, 4, 7, 5),     
        ],
    },
        
    "inside": {
        # ── overall dimensions ──────────────────────────────────────────────
        "size": (10.0, 10.0),          # W, H  in metres

        # ── fixed locations ────────────────────────────────────────────────
        "depot": (5.5, 6.0),           # lone pick-up point
        "delivery": (5.5, 2.5),      # lone drop-off point
        # "starting_position": (1.0, 1.0), # starting position of the robot

        # ── obstacles (axis-aligned boxes) ─────────────────────────────────
        #    xmin, ymin, xmax, ymax
        "obstacles": [
            (2.5, 1.2, 4.0, 8.6),      
            (1.0, 3.5, 9.5, 5.2),     
            (7.0, 1.2, 8.5, 8.6),      
        ],
    },
    

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