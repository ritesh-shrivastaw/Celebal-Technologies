"""
data/brochure_generator.py
Generates rich synthetic car brochure content for 4 brands × 3 models each.
In production, replace with real PDF brochures from manufacturer websites.
e.g. Hyundai: https://www.hyundai.com/in/en/find-a-car/brochure
"""

import os, json

BROCHURE_DATA = {
    "Hyundai": {
        "Creta": {
            "version": "2024",
            "sections": {
                "Overview": """
The Hyundai Creta 2024 is a feature-rich mid-size SUV available in petrol, diesel,
and turbo-petrol engine options. It is one of India's best-selling SUVs, known for
its bold design, premium interiors, and advanced technology features.
Available variants: E, EX, S, S(O), SX, SX Tech, SX(O), SX(O) Knight Edition.
Seating capacity: 5 passengers. Ground clearance: 190 mm. Body type: SUV.
Colours available: Atlas White, Abyss Black, Ranger Khaki, Robust Emerald,
Fiery Red, Typhoon Silver, Galaxy Grey, and two-tone options.
""",
                "Engine and Performance": """
Engine options for Hyundai Creta 2024:
1. 1.5L MPi Petrol (Smartstream): 115 PS @ 6300 rpm, 144 Nm @ 4500 rpm.
   Transmission: 6-speed iMT or 6-speed AT.
2. 1.5L CRDi Diesel (U2): 116 PS @ 4000 rpm, 250 Nm @ 1500-2750 rpm.
   Transmission: 6-speed MT or 6-speed AT.
3. 1.5L T-GDi Turbo Petrol: 160 PS @ 5500 rpm, 253 Nm @ 1500-3500 rpm.
   Transmission: 7-speed DCT.
Drive type: FWD (front-wheel drive) across all variants.
Turbo petrol offers the highest performance with 0-100 km/h in 8.9 seconds.
""",
                "Mileage and Fuel Efficiency": """
Hyundai Creta 2024 ARAI-certified mileage figures:
- 1.5L Petrol MT: 17.4 km/l
- 1.5L Petrol AT: 17.0 km/l
- 1.5L Petrol iMT: 17.4 km/l
- 1.5L Diesel MT: 21.8 km/l
- 1.5L Diesel AT: 18.5 km/l
- 1.5L Turbo Petrol DCT: 17.0 km/l
Fuel tank capacity: 50 litres across all variants.
Real-world mileage may vary based on driving conditions, traffic, and driving style.
The diesel variant offers the best fuel efficiency suitable for highway driving.
""",
                "Dimensions": """
Hyundai Creta 2024 dimensions:
Length: 4,330 mm | Width: 1,790 mm | Height: 1,635 mm
Wheelbase: 2,610 mm | Ground Clearance: 190 mm
Boot space: 433 litres (expandable with rear seats folded)
Kerb weight: 1,316 kg – 1,500 kg depending on variant.
Turning radius: 5.3 metres. Tyre size: 215/60 R17 (lower variants),
215/55 R17 (higher variants). Alloy wheel size: 17-inch.
""",
                "Safety": """
Hyundai Creta 2024 safety features:
- 6 airbags (standard across all variants from 2023 update)
- Electronic Stability Control (ESC)
- Vehicle Stability Management (VSM)
- Hill Assist Control (HAC)
- Rear Parking Sensors and Camera
- Blind Spot Collision Warning (BSCW) – SX and above
- Lane Keeping Assist (LKA) – SX(O) variant
- Forward Collision-Avoidance Assist (FCA) – SX(O) variant
- Driver Attention Warning (DAW) – SX(O) variant
- Tyre Pressure Monitoring System (TPMS) – SX(O) variant
- Auto-hold and EPB (Electronic Parking Brake) – SX and above
Global NCAP safety rating: 3 stars (adult), 4 stars (child).
""",
                "Interior and Comfort": """
Hyundai Creta 2024 interior and comfort features:
- 10.25-inch touchscreen infotainment display (SX and above)
- 10.25-inch digital instrument cluster (SX Tech and above)
- Bose 8-speaker premium sound system (SX(O) variant)
- Ventilated front seats (SX and above)
- Panoramic sunroof (SX(O) variant)
- Leatherette seat upholstery (SX and above)
- Dual-zone automatic climate control (SX Tech and above)
- Wireless phone charger (SX and above)
- 60:40 split-fold rear seats
- Rear AC vents and USB charging ports
- Ambient lighting (SX(O) variant)
- Electric sunroof (SX variant)
- Height-adjustable driver seat
""",
                "Infotainment and Connectivity": """
Hyundai Creta 2024 infotainment and connectivity:
- Hyundai BlueLink connected car technology (SX and above):
  Remote start/stop, live vehicle tracking, geo-fencing alerts,
  remote climate control, SOS emergency call, roadside assistance.
- Wireless Android Auto and Apple CarPlay (SX and above)
- OTA (Over-the-Air) map updates
- 10.25-inch touchscreen with split-screen capability
- Voice recognition (English and Hindi)
- 4 USB ports (2 front + 2 rear)
- Bluetooth multi-device connectivity
- JBL/Bose premium sound system options
- Amazon Alexa and Google Home integration (via BlueLink)
""",
                "Pricing": """
Hyundai Creta 2024 ex-showroom price range (approximate):
- E variant (Petrol MT): ₹11.00 lakh
- EX variant (Petrol MT): ₹13.00 lakh
- S variant (Petrol MT): ₹14.16 lakh
- S(O) variant (Petrol AT): ₹15.50 lakh
- SX variant (Turbo Petrol DCT): ₹17.99 lakh
- SX Tech (Turbo Petrol DCT): ₹18.79 lakh
- SX(O) variant (Turbo Petrol DCT): ₹20.45 lakh
- SX(O) Diesel AT: ₹20.45 lakh
Prices vary by city. On-road price includes RTO, insurance, and accessories.
"""
            }
        },
        "i20": {
            "version": "2024",
            "sections": {
                "Overview": """
The Hyundai i20 2024 is a premium hatchback offering a blend of sporty design,
feature-rich interior, and efficient engines. Available in petrol, diesel, and
turbo-petrol variants. Seating capacity: 5 passengers. Body type: Hatchback.
Variants: Era, Magna, Sportz, Sportz (O), Asta, Asta (O). Available in
6 colour options including Starry Night, Fiery Red, Titan Grey.
""",
                "Engine and Performance": """
Hyundai i20 2024 engine specifications:
1. 1.2L MPi Petrol (Kappa): 83 PS @ 6000 rpm, 114 Nm @ 4200 rpm.
   Transmission: 5-speed MT or IVT (Intelligent Variable Transmission).
2. 1.0L T-GDi Turbo Petrol: 120 PS @ 6000 rpm, 172 Nm @ 1500-4000 rpm.
   Transmission: 7-speed DCT or 6-speed iMT.
3. 1.5L CRDi Diesel: 100 PS @ 4000 rpm, 240 Nm @ 1500-2750 rpm.
   Transmission: 6-speed MT or 6-speed AT.
""",
                "Mileage and Fuel Efficiency": """
Hyundai i20 2024 ARAI-certified fuel efficiency:
- 1.2L Petrol MT: 20.35 km/l
- 1.2L Petrol IVT: 20.35 km/l
- 1.0L Turbo Petrol DCT: 20.25 km/l
- 1.0L Turbo Petrol iMT: 20.77 km/l
- 1.5L Diesel MT: 25.05 km/l
- 1.5L Diesel AT: 24.15 km/l
Fuel tank capacity: 37 litres.
The diesel variant delivers class-leading fuel efficiency.
""",
                "Safety": """
Hyundai i20 2024 safety features:
- 6 airbags (standard on all variants)
- ESC (Electronic Stability Control)
- Hill Start Assist (HSA)
- Rear parking sensors and camera
- ISOFIX child seat anchors
- Cornering brake control
- Blind Spot Collision Warning (Asta O)
- Lane Keeping Assist (Asta O)
""",
                "Dimensions": """
Hyundai i20 2024 dimensions:
Length: 3,995 mm | Width: 1,775 mm | Height: 1,505 mm
Wheelbase: 2,580 mm | Ground clearance: 161 mm
Boot space: 311 litres. Kerb weight: 950–1,115 kg.
Tyre size: 185/65 R15 (lower variants), 195/55 R16 (higher variants).
""",
                "Interior and Comfort": """
Hyundai i20 2024 interior features:
- 10.25-inch touchscreen (Sportz and above)
- Digital instrument cluster
- Bose 7-speaker sound system (Asta O)
- Electric sunroof (Sportz and above)
- Rear AC vents
- Wireless charging (Asta variant)
- Leatherette seats (Asta variant)
- Auto climate control (Asta variant)
""",
                "Infotainment and Connectivity": """
Hyundai i20 2024 connectivity:
- Hyundai BlueLink (Sportz and above)
- Wireless Android Auto and Apple CarPlay
- Voice recognition
- 4 USB charging ports
- Bluetooth connectivity
- OTA map updates
""",
                "Pricing": """
Hyundai i20 2024 price range:
- Era (1.2 Petrol MT): ₹7.04 lakh
- Magna (1.2 Petrol MT): ₹8.97 lakh
- Sportz (1.2 Petrol MT): ₹9.99 lakh
- Asta (1.0 Turbo DCT): ₹12.22 lakh
- Asta (O) (1.0 Turbo DCT): ₹13.22 lakh
Ex-showroom prices. On-road costs extra.
"""
            }
        },
        "Verna": {
            "version": "2024",
            "sections": {
                "Overview": """
The Hyundai Verna 2024 is a premium sedan with a bold fluidic sculpture design.
Available with petrol and turbo-petrol engines. 5-seater sedan.
Variants: EX, S, S(O), SX, SX Tech, SX(O). Competes with Honda City, Maruti Ciaz.
""",
                "Engine and Performance": """
Hyundai Verna 2024 engines:
1. 1.5L MPi Petrol: 115 PS, 144 Nm. 6-speed MT / IVT.
2. 1.5L T-GDi Turbo Petrol: 160 PS, 253 Nm. 7-speed DCT / 6-speed MT.
0-100 km/h: 8.5 seconds (turbo petrol DCT).
""",
                "Mileage and Fuel Efficiency": """
Hyundai Verna 2024 mileage:
- 1.5L Petrol MT: 18.6 km/l
- 1.5L Petrol IVT: 18.6 km/l
- 1.5L Turbo Petrol DCT: 17.7 km/l
- 1.5L Turbo Petrol MT: 18.2 km/l
Fuel tank: 45 litres.
""",
                "Safety": """
Hyundai Verna 2024 safety:
- 6 airbags standard
- ESC, VSM, HAC
- Forward Collision-Avoidance Assist
- Lane Keeping Assist
- Blind Spot Collision Warning
- Rear Cross-Traffic Collision Warning
- TPMS, Auto headlamps
- 360-degree camera (SX O)
Global NCAP: 6 stars (highest rating).
""",
                "Dimensions": """
Hyundai Verna 2024 dimensions:
Length: 4,535 mm | Width: 1,765 mm | Height: 1,475 mm
Wheelbase: 2,670 mm | Ground clearance: 145 mm
Boot space: 528 litres.
""",
                "Interior and Comfort": """
Hyundai Verna 2024 interior:
- 10.25-inch touchscreen
- 10.25-inch digital cluster
- Bose 8-speaker system
- Panoramic sunroof
- Ventilated and heated front seats
- Leatherette upholstery
- Dual-zone climate control
- Ambient lighting (64 colours)
""",
                "Infotainment and Connectivity": """
Hyundai Verna 2024 connectivity:
- BlueLink with 60+ connected features
- Wireless Android Auto / Apple CarPlay
- OTA updates
- Voice recognition (EN + HI)
""",
                "Pricing": """
Hyundai Verna 2024 price:
- EX: ₹10.90 lakh
- S: ₹13.29 lakh
- SX: ₹15.99 lakh
- SX(O) Turbo DCT: ₹17.99 lakh
"""
            }
        }
    },
    "Maruti Suzuki": {
        "Swift": {
            "version": "2024",
            "sections": {
                "Overview": """
The Maruti Suzuki Swift 2024 (4th generation) is a sporty hatchback with a
completely new design. Known for reliability, low running costs and high resale value.
5-seater hatchback. Variants: LXi, VXi, ZXi, ZXi+. Competes with Hyundai i20.
""",
                "Engine and Performance": """
Maruti Suzuki Swift 2024 engine:
- 1.2L Z-Series Petrol: 82 PS @ 5700 rpm, 112 Nm @ 4300 rpm.
- Transmissions: 5-speed MT or 5-speed AMT (Auto Gear Shift).
- No diesel variant available (discontinued).
0-100 km/h: 12.0 seconds (MT). CNG option available in VXi and ZXi variants.
""",
                "Mileage and Fuel Efficiency": """
Maruti Swift 2024 ARAI mileage:
- Petrol MT: 24.82 km/l (class-leading petrol mileage)
- Petrol AMT: 25.75 km/l
- CNG MT: 32.85 km/kg
Fuel tank: 37 litres (petrol), 60 litres equivalent (CNG).
Best-in-class fuel efficiency for a hatchback.
""",
                "Safety": """
Maruti Swift 2024 safety:
- 6 airbags (ZXi and ZXi+), 2 airbags (LXi, VXi)
- ESC standard across all variants
- Hill Hold Control
- Rear parking sensors and camera (ZXi and above)
- ISOFIX child seat anchors
- Seatbelt reminders for all seats
- ABS with EBD
Global NCAP: 3 stars.
""",
                "Dimensions": """
Maruti Swift 2024 dimensions:
Length: 3,860 mm | Width: 1,735 mm | Height: 1,530 mm
Wheelbase: 2,450 mm | Ground clearance: 163 mm
Boot space: 268 litres. Kerb weight: 895–925 kg (lightest in class).
Tyre: 185/65 R15.
""",
                "Interior and Comfort": """
Maruti Swift 2024 interior:
- 9-inch SmartPlay Pro+ touchscreen (ZXi and above)
- Digital instrument cluster with colour MID
- Rear AC vents
- Arkamys-tuned 6-speaker sound system (ZXi+)
- Auto climate control (ZXi+)
- Push button start (ZXi and above)
- Height adjustable driver seat
- Tilt and telescopic steering
""",
                "Infotainment and Connectivity": """
Maruti Swift 2024 connectivity:
- SmartPlay Pro+ with wireless Android Auto / Apple CarPlay
- Suzuki Connect (connected car features)
- Remote monitoring via app
- Voice commands
- Bluetooth multi-device
- 2 USB ports
""",
                "Pricing": """
Maruti Swift 2024 price:
- LXi: ₹6.49 lakh
- VXi: ₹7.39 lakh
- ZXi: ₹8.99 lakh
- ZXi+: ₹9.64 lakh
- ZXi AMT: ₹9.74 lakh
- ZXi+ AMT: ₹10.39 lakh
Ex-showroom Delhi.
"""
            }
        },
        "Brezza": {
            "version": "2024",
            "sections": {
                "Overview": """
The Maruti Suzuki Brezza 2024 is a compact SUV (sub-4-metre) with strong brand
trust and highest resale value in its segment. 5-seater SUV.
Variants: LXi, VXi, ZXi, ZXi+. Competes with Hyundai Venue, Kia Sonet.
""",
                "Engine and Performance": """
Maruti Brezza 2024 engine:
- 1.5L K15C Smart Hybrid Petrol: 103 PS, 137 Nm.
- Transmissions: 5-speed MT or 6-speed AT.
- CNG option: 88.5 PS, 121.5 Nm. 5-speed MT only.
Mild hybrid system gives small boost and improves efficiency.
""",
                "Mileage and Fuel Efficiency": """
Maruti Brezza 2024 ARAI fuel efficiency:
- Petrol MT: 19.89 km/l
- Petrol AT: 19.80 km/l
- CNG MT: 25.51 km/kg
Fuel tank: 48 litres.
""",
                "Safety": """
Maruti Brezza 2024 safety:
- 6 airbags (ZXi and above), 2 airbags (LXi/VXi)
- ESC, Hill Hold Control
- Rear parking sensors and camera
- 360-degree camera (ZXi+)
- TPMS (ZXi+)
- Auto headlamps
- ABS + EBD
""",
                "Dimensions": """
Maruti Brezza 2024 dimensions:
Length: 3,995 mm | Width: 1,790 mm | Height: 1,685 mm
Wheelbase: 2,500 mm | Ground clearance: 198 mm (highest in segment)
Boot space: 328 litres.
""",
                "Interior and Comfort": """
Maruti Brezza 2024 interior:
- 9-inch SmartPlay Pro+ touchscreen
- Head-up display (ZXi+)
- 360-degree camera display
- Wireless charging
- Ventilated front seats (ZXi+)
- Sunroof (ZXi and above)
- Auto climate control
- 6-speaker sound system
""",
                "Infotainment and Connectivity": """
Maruti Brezza 2024 connectivity:
- Suzuki Connect with 40+ features
- Wireless Android Auto / Apple CarPlay
- Voice commands
- Remote start/stop (AT only via app)
- OTA map updates
""",
                "Pricing": """
Maruti Brezza 2024 price:
- LXi: ₹8.34 lakh
- VXi: ₹9.99 lakh
- ZXi: ₹12.34 lakh
- ZXi+: ₹13.96 lakh
- ZXi+ AT: ₹15.05 lakh
Ex-showroom prices.
"""
            }
        },
        "Ertiga": {
            "version": "2024",
            "sections": {
                "Overview": """
Maruti Suzuki Ertiga 2024 is a 7-seater MPV known for practicality, comfort,
and low running costs. Most popular MPV in India. Variants: LXi, VXi, ZXi, ZXi+.
Competes with Kia Carens, Hyundai Alcazar.
""",
                "Engine and Performance": """
Maruti Ertiga 2024 engine:
- 1.5L K15C Smart Hybrid Petrol: 103 PS, 137 Nm.
- Transmissions: 5-speed MT or 6-speed AT.
- CNG option: 87.8 PS. 5-speed MT only.
""",
                "Mileage and Fuel Efficiency": """
Maruti Ertiga 2024 mileage:
- Petrol MT: 20.51 km/l
- Petrol AT: 20.30 km/l
- CNG MT: 26.11 km/kg
Fuel tank: 45 litres.
""",
                "Safety": """
Maruti Ertiga 2024 safety:
- 6 airbags (ZXi and above)
- ESC
- Rear parking camera
- ABS + EBD
- Hill Hold Control
- ISOFIX child anchors
""",
                "Dimensions": """
Maruti Ertiga 2024 dimensions:
Length: 4,395 mm | Width: 1,735 mm | Height: 1,690 mm
Wheelbase: 2,740 mm | Ground clearance: 180 mm
Boot space: 135 litres (3rd row up), 550 litres (3rd row folded).
Seating: 7 passengers across 3 rows.
""",
                "Interior and Comfort": """
Maruti Ertiga 2024 interior:
- 7-inch SmartPlay Studio touchscreen (upgraded to 9-inch on ZXi)
- Captain seats option for 6-seater config
- Rear reading lamps
- Auto climate control (ZXi)
- USB charging ports (all 3 rows)
- 60:40 foldable 3rd row seats
""",
                "Infotainment and Connectivity": """
Maruti Ertiga 2024 connectivity:
- SmartPlay Pro+ (ZXi and above)
- Android Auto / Apple CarPlay
- Suzuki Connect
- Bluetooth and voice commands
""",
                "Pricing": """
Maruti Ertiga 2024 price:
- LXi: ₹8.69 lakh
- VXi: ₹10.49 lakh
- ZXi: ₹12.29 lakh
- ZXi+: ₹12.74 lakh
- ZXi+ AT: ₹14.05 lakh
"""
            }
        }
    },
    "Tata": {
        "Nexon": {
            "version": "2024",
            "sections": {
                "Overview": """
Tata Nexon 2024 is India's first 5-star Global NCAP rated car. Sub-4-metre SUV
available in petrol, diesel, and electric (Nexon EV) variants.
5-seater. Variants: Smart, Smart+, Pure, Pure+, Creative, Creative+, Fearless, Fearless+.
Competes with Maruti Brezza, Hyundai Venue, Kia Sonet.
""",
                "Engine and Performance": """
Tata Nexon 2024 engine options:
1. 1.2L Revotron Turbo Petrol: 120 PS @ 5000 rpm, 170 Nm @ 2000-3500 rpm.
   Transmissions: 6-speed MT, 6-speed AMT, or 7-speed DCT.
2. 1.5L Revotorq Diesel: 115 PS @ 4000 rpm, 260 Nm @ 1500-2750 rpm.
   Transmissions: 6-speed MT or 6-speed AMT.
Also available: Nexon EV (electric) — separate brochure.
""",
                "Mileage and Fuel Efficiency": """
Tata Nexon 2024 ARAI mileage:
- Petrol MT: 17.01 km/l
- Petrol AMT: 17.05 km/l
- Petrol DCT: 18.23 km/l
- Diesel MT: 23.23 km/l
- Diesel AMT: 22.37 km/l
Fuel tank: 44 litres.
Diesel offers best real-world efficiency for long-distance driving.
""",
                "Safety": """
Tata Nexon 2024 safety — India's first 5-star Global NCAP car:
- 6 airbags standard (all variants)
- ESC, Rollover Mitigation
- Corner Stability Control
- Emergency Brake Assist
- ABS + EBD
- Electronic Brake force Distribution
- Hill Hold and Descent Control
- Rear parking camera and sensors
- TPMS
- Blind Spot Detection (Fearless+ and above)
- Auto-dimming IRVM (Fearless+)
Global NCAP: 5 stars adult, 3 stars child.
""",
                "Dimensions": """
Tata Nexon 2024 dimensions:
Length: 3,994 mm | Width: 1,811 mm | Height: 1,620 mm
Wheelbase: 2,498 mm | Ground clearance: 208 mm (highest in segment)
Boot space: 350 litres. Kerb weight: 1,190–1,320 kg.
""",
                "Interior and Comfort": """
Tata Nexon 2024 interior:
- 12.3-inch floating touchscreen (Fearless and above)
- 10.25-inch digital instrument cluster
- JBL 9-speaker premium sound system (Fearless+)
- Panoramic sunroof (Fearless and above)
- Ventilated front seats (Fearless+)
- Leatherette upholstery (Creative and above)
- Air purifier with AQI display
- Wireless charging
- Ambient lighting (Fearless)
- 6-way power adjustable driver seat
""",
                "Infotainment and Connectivity": """
Tata Nexon 2024 connectivity:
- Tata iRA connected car technology with 50+ features
- Wireless Android Auto / Apple CarPlay
- OTA software updates
- Voice commands (Alexa and Google Assistant built-in)
- Remote AC control, live tracking, trip analysis
- 5 USB ports (3 Type-A + 2 Type-C)
""",
                "Pricing": """
Tata Nexon 2024 price range:
- Smart (Petrol MT): ₹8.10 lakh
- Pure (Petrol MT): ₹9.30 lakh
- Creative (Petrol DCT): ₹13.49 lakh
- Fearless (Petrol DCT): ₹14.99 lakh
- Fearless+ (Diesel AMT): ₹16.49 lakh
Ex-showroom prices. EV variants priced separately.
"""
            }
        },
        "Harrier": {
            "version": "2024",
            "sections": {
                "Overview": """
Tata Harrier 2024 is a premium 5-seater SUV based on Omega Arc platform
derived from Land Rover's D8 architecture. Bold design, premium features.
Competes with Hyundai Tucson, MG Hector, Jeep Compass.
Variants: Smart, Smart+, Pure+, Adventure, Adventure+, Creative, Creative+, Fearless, Fearless+.
""",
                "Engine and Performance": """
Tata Harrier 2024 engine:
- 2.0L Kryotec Diesel: 170 PS @ 3750 rpm, 350 Nm @ 1750-2500 rpm.
- Transmissions: 6-speed MT or 6-speed AT (Hyundai-sourced).
- Petrol engine expected 2024 (announcement pending).
Drive modes: Eco, City, Sport. Terrain modes: Normal, Wet, Rough.
Tow capacity: 2500 kg (with tow pack).
""",
                "Mileage and Fuel Efficiency": """
Tata Harrier 2024 mileage:
- Diesel MT: 16.35 km/l
- Diesel AT: 14.64 km/l
Fuel tank: 50 litres.
""",
                "Safety": """
Tata Harrier 2024 safety:
- 6 airbags standard
- ADAS Level 2 features (Fearless+):
  Autonomous Emergency Braking, Adaptive Cruise Control,
  Lane Departure Warning, Lane Keep Assist, Traffic Sign Recognition
- Blind Spot Detection
- 360-degree surround view camera
- Electronic Stability Program
- Hill Descent Control
- Emergency Stop Signal
Global NCAP: 5 stars.
""",
                "Dimensions": """
Tata Harrier 2024 dimensions:
Length: 4,598 mm | Width: 1,894 mm | Height: 1,706 mm
Wheelbase: 2,741 mm | Ground clearance: 205 mm
Boot space: 425 litres. Kerb weight: 1,675–1,750 kg.
""",
                "Interior and Comfort": """
Tata Harrier 2024 interior:
- 12.3-inch floating touchscreen
- 10.25-inch digital instrument cluster
- JBL 9-speaker Meridian sound system (top variants)
- Panoramic sunroof
- Ventilated and heated front seats
- Leather upholstery
- Dual-zone automatic climate control
- Ambient lighting (64 colours)
- Electric sunshade for rear windscreen
""",
                "Infotainment and Connectivity": """
Tata Harrier 2024 connectivity:
- Tata iRA with 70+ connected features
- OTA updates
- Wireless Android Auto / Apple CarPlay
- Alexa built-in
- Remote start/stop, AC control
""",
                "Pricing": """
Tata Harrier 2024 price:
- Smart: ₹15.49 lakh
- Adventure: ₹19.49 lakh
- Creative: ₹22.49 lakh
- Fearless: ₹24.49 lakh
- Fearless+ AT: ₹26.44 lakh
"""
            }
        },
        "Punch": {
            "version": "2024",
            "sections": {
                "Overview": """
Tata Punch 2024 is a micro SUV positioned below the Nexon. India's safest micro SUV.
5-seater. Variants: Pure, Adventure, Accomplished, Creative.
Competes with Maruti Ignis, Citroen C3.
""",
                "Engine and Performance": """
Tata Punch 2024 engine:
- 1.2L Revotron Petrol: 86 PS @ 6000 rpm, 113 Nm @ 3300 rpm.
- Transmissions: 5-speed MT or 5-speed AMT.
- CNG option: 73.5 PS. 5-speed MT only.
""",
                "Mileage and Fuel Efficiency": """
Tata Punch 2024 mileage:
- Petrol MT: 18.82 km/l
- Petrol AMT: 18.82 km/l
- CNG: 26.99 km/kg
Fuel tank: 37 litres.
""",
                "Safety": """
Tata Punch 2024 safety:
- 2 airbags standard, 4 airbags (Creative)
- Global NCAP: 5 stars — safest micro SUV in India
- ESC, Corner Stability Control
- ABS + EBD
- Hill Hold Control
- Rear parking camera and sensors
""",
                "Dimensions": """
Tata Punch 2024 dimensions:
Length: 3,827 mm | Width: 1,742 mm | Height: 1,615 mm
Wheelbase: 2,445 mm | Ground clearance: 187 mm
Boot space: 366 litres.
""",
                "Interior and Comfort": """
Tata Punch 2024 interior:
- 7-inch touchscreen (Adventure and above)
- Semi-digital instrument cluster
- Electric sunroof (Accomplished and above)
- Auto AC (Creative)
- 4-speaker sound system
- USB charging ports
""",
                "Infotainment and Connectivity": """
Tata Punch 2024 connectivity:
- iRA connected car (Creative variant)
- Android Auto / Apple CarPlay
- Bluetooth and voice commands
""",
                "Pricing": """
Tata Punch 2024 price:
- Pure: ₹6.00 lakh
- Adventure: ₹7.49 lakh
- Accomplished: ₹9.49 lakh
- Creative AMT: ₹10.99 lakh
"""
            }
        }
    },
    "Mahindra": {
        "Scorpio-N": {
            "version": "2024",
            "sections": {
                "Overview": """
Mahindra Scorpio-N is a body-on-frame SUV — the new generation Scorpio.
Available in 6 and 7 seater configurations. Petrol and diesel engines.
Variants: Z2, Z4, Z6, Z8, Z8 L. True body-on-frame SUV for off-road enthusiasts.
Competes with Toyota Fortuner, Tata Harrier, MG Hector Plus.
""",
                "Engine and Performance": """
Mahindra Scorpio-N 2024 engines:
1. 2.0L mStallion Turbo Petrol: 200 PS @ 5000 rpm, 380 Nm @ 1750-3000 rpm.
   AWD available. Transmissions: 6-speed MT or 6-speed AT.
2. 2.2L mHawk Diesel: 175 PS @ 3500 rpm, 400 Nm @ 1500-2800 rpm.
   AWD available on Z6 and above. Transmissions: 6-speed MT or 6-speed AT.
4XPLOR AWD system with intelligent torque distribution.
Wading depth: 550 mm. Approach angle: 30.2 degrees.
""",
                "Mileage and Fuel Efficiency": """
Mahindra Scorpio-N 2024 mileage:
- Diesel MT: 15.21 km/l
- Diesel AT: 14.06 km/l
- Petrol MT: 13.69 km/l
- Petrol AT: 13.21 km/l
Fuel tank: 60 litres (largest in segment).
""",
                "Safety": """
Mahindra Scorpio-N 2024 safety:
- 6 airbags (Z6 and above), 2 airbags (Z2, Z4)
- ESC, Rollover Mitigation, HSA
- ABS + EBD
- Rear disc brakes (Z6 and above)
- Reverse camera
- Blind Spot Detection (Z8 L)
- Electronic Brake Distribution
Global NCAP: 5 stars.
""",
                "Dimensions": """
Mahindra Scorpio-N 2024 dimensions:
Length: 4,662 mm | Width: 1,917 mm | Height: 1,857 mm
Wheelbase: 2,750 mm | Ground clearance: 200 mm
Boot space: 289 litres (7 seater, 3rd row up), 674 litres (3rd row folded).
Seating: 6 or 7 passengers.
""",
                "Interior and Comfort": """
Mahindra Scorpio-N 2024 interior:
- 8-inch infotainment touchscreen (Z4 and above)
- 12.3-inch Sony 3D surround sound system (Z8 L)
- Wireless charging (Z6 and above)
- Sunroof (Z4 and above)
- Ventilated front seats (Z8 and above)
- Leather upholstery (Z6 and above)
- Dual-zone climate control (Z8)
- Third row seats (7-seater config)
""",
                "Infotainment and Connectivity": """
Mahindra Scorpio-N 2024 connectivity:
- AdrenoX connected car platform
- Wireless Android Auto / Apple CarPlay
- Alexa and Google Assistant
- Remote start/stop, tracking
- OTA updates
- 5G ready hardware
""",
                "Pricing": """
Mahindra Scorpio-N 2024 price:
- Z2 (Diesel MT): ₹13.99 lakh
- Z4 (Diesel MT): ₹15.49 lakh
- Z6 (Diesel MT): ₹17.49 lakh
- Z8 (Diesel MT): ₹20.99 lakh
- Z8 L (Diesel AT, AWD): ₹24.99 lakh
- Z8 L (Petrol AT, AWD): ₹23.99 lakh
"""
            }
        },
        "XUV 3XO": {
            "version": "2024",
            "sections": {
                "Overview": """
Mahindra XUV 3XO (previously XUV300) is a feature-loaded sub-4m compact SUV.
Completely redesigned for 2024 with new powertrains and ADAS features.
5-seater. Variants: MX1, MX2, MX2 Pro, AX5, AX5 L, AX7, AX7 L.
Competes with Tata Nexon, Hyundai Venue, Kia Sonet.
""",
                "Engine and Performance": """
Mahindra XUV 3XO 2024 engines:
1. 1.2L TGDi Turbo Petrol: 130 PS, 230 Nm. 6-speed MT or 6-speed AT.
2. 1.2L mFalcon Turbo Petrol: 110 PS, 200 Nm. 6-speed MT or 6-speed AMT.
3. 1.5L mHawk Diesel: 117 PS, 300 Nm. 6-speed MT or 6-speed AT.
""",
                "Mileage and Fuel Efficiency": """
Mahindra XUV 3XO 2024 mileage:
- TGDi Petrol AT: 18.31 km/l
- Turbo Petrol MT: 20.07 km/l
- Diesel MT: 20.15 km/l
- Diesel AT: 20.07 km/l
Fuel tank: 40 litres.
""",
                "Safety": """
Mahindra XUV 3XO 2024 safety:
- 6 airbags standard
- ADAS Level 2 (AX5 L and above):
  AEB, Adaptive Cruise Control, Lane Keep Assist, Traffic Sign Recognition
- ESC, Rollover Mitigation
- 360-degree camera
- Rear disc brakes
Global NCAP: 5 stars (XUV300 predecessor).
""",
                "Dimensions": """
Mahindra XUV 3XO 2024 dimensions:
Length: 3,990 mm | Width: 1,821 mm | Height: 1,647 mm
Wheelbase: 2,600 mm | Ground clearance: 201 mm
Boot space: 364 litres.
""",
                "Interior and Comfort": """
Mahindra XUV 3XO 2024 interior:
- 10.25-inch touchscreen (AX5 and above)
- 10.25-inch digital cluster
- Panoramic sunroof (AX7)
- Wireless charging
- Ventilated seats (AX7 L)
- Harman Kardon 9-speaker audio (AX7 L)
- Dual-zone auto climate control
""",
                "Infotainment and Connectivity": """
Mahindra XUV 3XO 2024 connectivity:
- AdrenoX with wireless Android Auto / Apple CarPlay
- Alexa and Google Assistant built-in
- OTA updates
- Remote features via app
""",
                "Pricing": """
Mahindra XUV 3XO 2024 price:
- MX1 (Petrol MT): ₹7.49 lakh
- MX2 Pro (Petrol MT): ₹9.99 lakh
- AX5 (Petrol AT): ₹12.49 lakh
- AX7 (Diesel MT): ₹14.99 lakh
- AX7 L (Diesel AT): ₹15.49 lakh
"""
            }
        },
        "Thar": {
            "version": "2024",
            "sections": {
                "Overview": """
Mahindra Thar 2024 (Thar ROXX) is an iconic lifestyle off-road SUV.
Now available in 5-door configuration. 4-seater (3-door) and 5-seater (5-door).
Variants: MX1, MX3, MX5, AX3L, AX5L, AX7L.
Competes with Force Gurkha, Jeep Wrangler.
""",
                "Engine and Performance": """
Mahindra Thar 2024 (ROXX) engines:
1. 2.2L mHawk Diesel: 175 PS, 370 Nm. 6-speed MT or AT. 4WD.
2. 2.0L mStallion Petrol: 175 PS, 300 Nm. 6-speed MT or AT. 4WD.
4×4 with low-range transfer case. Wading depth: 650 mm.
Approach angle: 41.8°. Departure angle: 36.4°.
""",
                "Mileage and Fuel Efficiency": """
Mahindra Thar ROXX mileage:
- Diesel MT: 16.42 km/l
- Diesel AT: 15.22 km/l
- Petrol MT: 14.21 km/l
Fuel tank: 57 litres.
""",
                "Safety": """
Mahindra Thar 2024 safety:
- 6 airbags (AX variants)
- ESC
- ABS + EBD
- Hill Descent Control
- Roll-over mitigation
- Rear parking camera
Global NCAP: Not rated (body-on-frame off-road class).
""",
                "Dimensions": """
Mahindra Thar ROXX (5-door) dimensions:
Length: 4,428 mm | Width: 1,870 mm | Height: 1,923 mm
Wheelbase: 2,850 mm | Ground clearance: 226 mm
Boot space: 447 litres.
""",
                "Interior and Comfort": """
Mahindra Thar 2024 interior:
- 12.3-inch curved touchscreen (AX variants)
- 10.25-inch digital cluster
- Wireless charging
- Ventilated seats (AX7 L)
- Convertible soft-top or hardtop (3-door)
- Waterproof interior
- Premium audio system
""",
                "Infotainment and Connectivity": """
Mahindra Thar 2024 connectivity:
- AdrenoX platform
- Wireless Android Auto / Apple CarPlay
- Alexa and Google built-in
- Remote features
- OTA updates
""",
                "Pricing": """
Mahindra Thar ROXX 2024 price:
- MX1 (Diesel MT, 5-door): ₹12.99 lakh
- MX3 (Petrol MT, 5-door): ₹14.99 lakh
- AX5 L (Diesel AT, 5-door): ₹20.49 lakh
- AX7 L (Petrol AT, 5-door): ₹22.49 lakh
Thar 3-door prices start from ₹11.35 lakh.
"""
            }
        }
    }
}


def save_brochures(output_dir: str):
    """Save all brochure data as structured JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    index = []
    for brand, models in BROCHURE_DATA.items():
        brand_dir = os.path.join(output_dir, brand.replace(" ", "_"))
        os.makedirs(brand_dir, exist_ok=True)
        for model, data in models.items():
            fname = f"{model.replace(' ', '_')}.json"
            fpath = os.path.join(brand_dir, fname)
            payload = {"brand": brand, "model": model, **data}
            with open(fpath, "w") as f:
                json.dump(payload, f, indent=2)
            index.append({"brand": brand, "model": model, "file": fpath})
            print(f"  ✓ {brand} {model} → {fpath}")
    with open(os.path.join(output_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"\n  ✓ Index saved. Total: {len(index)} brochures.")
    return index


def get_brands():
    return list(BROCHURE_DATA.keys())


def get_models(brand: str):
    return list(BROCHURE_DATA.get(brand, {}).keys())


if __name__ == "__main__":
    print("\n[DriveWise] Generating synthetic car brochures...")
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_brochures(os.path.join(ROOT, "brochures"))
    print("\n  Done. 4 brands × 3 models = 12 brochures.\n")