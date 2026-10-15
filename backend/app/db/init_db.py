from datetime import datetime, timezone, timedelta
from app.db.session import engine, SessionLocal
from app.db.base import Base
import app.models  # Ensures all models are registered
from app.models.user import User, UserRole
from app.models.road_memory import HotspotCluster
from app.models.incident import RoadSegment, RoadIncident
from app.core.security import hash_password


def init_database(seed_initial_data: bool = True, engine_bind=None, session_maker=None):
    """Creates all database tables and seeds initial administrative users and demo segments."""
    eng = engine_bind or engine
    sm = session_maker or SessionLocal
    Base.metadata.create_all(bind=eng)
    print("[RoadSense DB] Tables created successfully.")

    if not seed_initial_data:
        return

    db = sm()

    try:
        # Check if users already exist
        admin_user = db.query(User).filter(User.username == "admin").first()
        if not admin_user:
            admin_user = User(
                username="admin",
                email="admin@roadsense.ai",
                hashed_password=hash_password("admin123"),
                full_name="Municipal Administrator",
                role=UserRole.ADMIN.value,
                is_active=True,
            )
            db.add(admin_user)

        driver_user = db.query(User).filter(User.username == "driver1").first()
        if not driver_user:
            driver_user = User(
                username="driver1",
                email="driver1@roadsense.ai",
                hashed_password=hash_password("driver123"),
                full_name="Field Telemetry Driver",
                role=UserRole.DRIVER.value,
                is_active=True,
            )
            db.add(driver_user)

        authority_user = db.query(User).filter(User.username == "authority1").first()
        if not authority_user:
            authority_user = User(
                username="authority1",
                email="inspector@pwd.delhi.gov.in",
                hashed_password=hash_password("authority123"),
                full_name="PWD Senior Highway Inspector",
                role=UserRole.AUTHORITY.value,
                is_active=True,
            )
            db.add(authority_user)

        # Seed sample road segments if none
        if db.query(RoadSegment).count() == 0:
            seg1 = RoadSegment(
                segment_name="Corridor A-1 (Connaught Place to Ring Road)",
                road_name="Barakhamba Road",
                city="New Delhi",
                district="Central",
                start_lat=28.6139,
                start_lon=77.2090,
                end_lat=28.6250,
                end_lon=77.2200,
                surface_type="Asphalt Bitumen",
                length_meters=1850.0,
            )
            seg2 = RoadSegment(
                segment_name="Ring Road Outer Arterial Section 4",
                road_name="Mahatma Gandhi Road",
                city="New Delhi",
                district="South",
                start_lat=28.6200,
                start_lon=77.2150,
                end_lat=28.6350,
                end_lon=77.2300,
                surface_type="Concrete Pavement",
                length_meters=2400.0,
            )
            db.add_all([seg1, seg2])

        # Seed realistic road incidents for Chhatrapati Sambhajinagar if none
        if db.query(RoadIncident).count() == 0:
            now = datetime.now(timezone.utc)
            aurangabad_seed = [
                {"title": "Deep Pothole Crater near Seven Hills Flyover", "distress_type": "Pothole", "severity": "Critical", "status": "reported", "conf": 0.96, "lat": 19.87698, "lon": 75.35380, "addr": "Jalna Road (NH-753F) near Seven Hills Flyover, Ward 02", "notes": "Z-shock spike: 29.8 m/s², Jerk: 245 m/s³, Depth: ~75mm."},
                {"title": "Unmarked Speed Bump near Mondha Naka", "distress_type": "Bump", "severity": "Medium", "status": "assigned", "conf": 0.91, "lat": 19.87648, "lon": 75.34750, "addr": "Jalna Road (NH-753F) at Mondha Naka, Ward 02", "assigned_to": "CSMC Rapid Response Squad 1 (Kranti Chowk Depot)", "notes": "Asphalt speed breaker crest. Z-impulse: 16.2 m/s²."},
                {"title": "Pothole Pit opposite Akashwani Circle", "distress_type": "Pothole", "severity": "High", "status": "under_review", "conf": 0.94, "lat": 19.87595, "lon": 75.34120, "addr": "Jalna Road, Opposite Akashwani Prasar Bharati, Ward 02", "notes": "Pavement cavity in outer right lane. Z-shock: 24.1 m/s²."},
                {"title": "Speed Bump before District Sessions Court", "distress_type": "Bump", "severity": "Low", "status": "resolved", "conf": 0.88, "lat": 19.87520, "lon": 75.33380, "addr": "Sessions Court / Jalna Road Cross, Ward 02", "assigned_to": "PWD Aurangabad Division Team Alpha", "notes": "Standard pedestrian crossing bump."},
                {"title": "Hazardous Pothole at Kranti Chowk Junction", "distress_type": "Pothole", "severity": "Critical", "status": "reported", "conf": 0.97, "lat": 19.87372, "lon": 75.32485, "addr": "Kranti Chowk Public Junction, Ward 01", "notes": "Major pothole at busy intersection. Z-shock: 32.1 m/s², Freefall dip: 2.8 m/s²."},
                {"title": "Speed Breaker at Gopal Tea / Nirala Bazar Link", "distress_type": "Bump", "severity": "Medium", "status": "scheduled", "conf": 0.89, "lat": 19.87445, "lon": 75.32890, "addr": "Gopal Tea Link, Nirala Bazar Road, Ward 01", "assigned_to": "CSMC Rapid Response Squad 1 (Kranti Chowk Depot)", "notes": "Sharp asphalt bump causing vehicle bottoming."},
                {"title": "Cavity Depression near Samarth Nagar Cross", "distress_type": "Pothole", "severity": "High", "status": "assigned", "conf": 0.92, "lat": 19.87250, "lon": 75.32100, "addr": "Samarth Nagar Link Road, Ward 01", "assigned_to": "PWD Aurangabad Division Team Alpha", "notes": "Subsurface sinkhole forming. Z-shock: 26.5 m/s²."},
                {"title": "Pothole Cluster at CIDCO Cannaught Place Entrance", "distress_type": "Pothole", "severity": "High", "status": "assigned", "conf": 0.93, "lat": 19.87785, "lon": 75.36250, "addr": "CIDCO Cannaught Place Entrance, Ward 03", "assigned_to": "ASCDCL Smart City Highway Squad", "notes": "Multiple pothole shocks within 8 meters. Z-shock: 25.4 m/s²."},
                {"title": "Dual Speed Hump near CIDCO Bus Stand", "distress_type": "Bump", "severity": "Medium", "status": "reported", "conf": 0.90, "lat": 19.87750, "lon": 75.35920, "addr": "CIDCO Town Centre Cross, Ward 03", "notes": "Double speed bumps installed without warning sign."},
                {"title": "Pothole in Front of APMC Market Gate", "distress_type": "Pothole", "severity": "Critical", "status": "under_review", "conf": 0.95, "lat": 19.87815, "lon": 75.36680, "addr": "APMC Market / Jalna Road, Ward 03", "notes": "Heavy truck impact crater. Z-shock: 31.0 m/s², Depth: ~85mm."},
                {"title": "Pothole Trench near MGM Medical College", "distress_type": "Pothole", "severity": "Critical", "status": "reported", "conf": 0.96, "lat": 19.87855, "lon": 75.37120, "addr": "MGM Medical College & Hospital Frontage, Ward 04", "notes": "Severe pothole on ambulance transit route. Z-shock: 28.7 m/s²."},
                {"title": "Speed Bump before Dhoot Hospital Junction", "distress_type": "Bump", "severity": "Medium", "status": "assigned", "conf": 0.92, "lat": 19.87910, "lon": 75.37620, "addr": "Dhoot Hospital Road Link, Ward 04", "assigned_to": "Contractor: Shri Ganesh Infra Roadways", "notes": "Hospital zone calming bump."},
                {"title": "Highway Pothole on Beed Bypass NH-52", "distress_type": "Pothole", "severity": "Critical", "status": "reported", "conf": 0.97, "lat": 19.84500, "lon": 75.34000, "addr": "Beed Bypass NH-52 Arterial, Ward 05", "notes": "Dangerous high-speed pothole on bypass. Z-shock: 33.5 m/s²."},
                {"title": "Rumble Strip Series near Mahanubhav Chowk", "distress_type": "Bump", "severity": "Low", "status": "resolved", "conf": 0.87, "lat": 19.84200, "lon": 75.33500, "addr": "Mahanubhav Chowk, Beed Bypass, Ward 05", "assigned_to": "PWD Aurangabad Division Team Alpha", "notes": "Traffic calming series before intersection."},
            ]
            for idx, item in enumerate(aurangabad_seed):
                inc = RoadIncident(
                    title=item["title"],
                    distress_type=item["distress_type"],
                    severity=item["severity"],
                    status=item["status"],
                    verification_status="sensor_confirmed",
                    confidence=item["conf"],
                    latitude=item["lat"],
                    longitude=item["lon"],
                    is_location_pending=False,
                    address_text=item["addr"],
                    assigned_to=item.get("assigned_to"),
                    notes=item["notes"],
                    created_at=now - timedelta(hours=idx * 2 + 1),
                )
                db.add(inc)

        db.commit()
        print("[RoadSense DB] Seeded initial users, segments, and incidents.")
    except Exception as e:
        db.rollback()
        print(f"[RoadSense DB] Seeding error: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    init_database()
