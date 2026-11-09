"""
Generador de datos sintéticos para SimSys Local.

Crea dos datasets de negocios distintos (~10.000 registros cada uno):

  - data/clientes.xlsx : clientes de una plataforma online de apuestas/casino
  - data/locales.xlsx  : locales físicos con máquinas de juego (hostelería, salones, bingos)

Cada registro se genera a partir de un PERFIL LATENTE (arquetipo) con sus propias
distribuciones, más ruido y valores nulos. La columna SEGMENTO_SIMULADO guarda ese
perfil: NO se usa para calcular similitudes (se excluye en config.py), solo para
VALIDAR que el sistema agrupa como cabría esperar.

Uso:
    python generar_datos.py                  # 10.000 + 10.000 registros
    python generar_datos.py --n-clientes 5000 --n-locales 3000 --seed 7
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

# ----------------------------------------------------------------------------
# Catálogos compartidos
# ----------------------------------------------------------------------------
PROVINCIAS = {
    # provincia: (comunidad, peso poblacional aprox., es_costa_turistica)
    "Madrid": ("Madrid", 14.0, False),
    "Barcelona": ("Cataluña", 12.0, True),
    "Valencia": ("C. Valenciana", 5.5, True),
    "Alicante": ("C. Valenciana", 4.0, True),
    "Sevilla": ("Andalucía", 4.1, False),
    "Málaga": ("Andalucía", 3.6, True),
    "Murcia": ("Murcia", 3.2, True),
    "Cádiz": ("Andalucía", 2.6, True),
    "Vizcaya": ("País Vasco", 2.4, False),
    "Baleares": ("Baleares", 2.5, True),
    "Las Palmas": ("Canarias", 2.3, True),
    "A Coruña": ("Galicia", 2.4, False),
    "Asturias": ("Asturias", 2.1, False),
    "Zaragoza": ("Aragón", 2.1, False),
    "Tarragona": ("Cataluña", 1.8, True),
    "Girona": ("Cataluña", 1.6, True),
    "Granada": ("Andalucía", 1.9, False),
    "Toledo": ("Castilla-La Mancha", 1.5, False),
    "Valladolid": ("Castilla y León", 1.1, False),
    "Badajoz": ("Extremadura", 1.4, False),
}
PROV_NOMBRES = list(PROVINCIAS)
PROV_PESOS = np.array([v[1] for v in PROVINCIAS.values()])
PROV_PESOS = PROV_PESOS / PROV_PESOS.sum()


def _lognormal(rng, media, sigma, size):
    """Lognormal parametrizada por la media deseada (más intuitivo que mu)."""
    mu = np.log(max(media, 1e-6)) - sigma ** 2 / 2
    return rng.lognormal(mu, sigma, size)


def _choice(rng, opciones, probs, size):
    probs = np.asarray(probs, dtype=float)
    return rng.choice(opciones, size=size, p=probs / probs.sum())


def _inyectar_nulos(rng, df, columnas_pct):
    for col, pct in columnas_pct.items():
        mask = rng.random(len(df)) < pct
        df.loc[mask, col] = np.nan
    return df


# ============================================================================
# CLIENTES (plataforma online de apuestas / casino)
# ============================================================================
PERFILES_CLIENTES = {
    # nombre: peso, edad, producto favorito (Sports, Casino, Poker), actividad (accesos/30d),
    #         depósito medio 30d (€), recencia media (días), antigüedad media (días), live %
    "Apostador deportivo casual": dict(peso=0.24, edad=(34, 9), prod=(0.85, 0.12, 0.03), acc=6, dep=45, rec=12, ant=900, live=0.25),
    "Apostador deportivo intensivo": dict(peso=0.10, edad=(29, 6), prod=(0.92, 0.06, 0.02), acc=26, dep=420, rec=2, ant=1400, live=0.65),
    "Jugador de slots": dict(peso=0.16, edad=(45, 11), prod=(0.05, 0.93, 0.02), acc=14, dep=160, rec=4, ant=1100, live=0.02),
    "Casino live alto valor": dict(peso=0.04, edad=(48, 9), prod=(0.08, 0.90, 0.02), acc=22, dep=2600, rec=2, ant=2200, live=0.05),
    "Pokerista": dict(peso=0.05, edad=(33, 8), prod=(0.08, 0.10, 0.82), acc=18, dep=210, rec=3, ant=1800, live=0.05),
    "Multiproducto": dict(peso=0.09, edad=(37, 9), prod=(0.45, 0.45, 0.10), acc=16, dep=300, rec=3, ant=1600, live=0.35),
    "Cliente dormido": dict(peso=0.20, edad=(40, 12), prod=(0.55, 0.38, 0.07), acc=0.4, dep=3, rec=160, ant=1900, live=0.15),
    "Nuevo registro": dict(peso=0.12, edad=(27, 7), prod=(0.70, 0.27, 0.03), acc=5, dep=35, rec=6, ant=35, live=0.20),
}


def generar_clientes(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    nombres = list(PERFILES_CLIENTES)
    pesos = [PERFILES_CLIENTES[p]["peso"] for p in nombres]
    perfiles = _choice(rng, nombres, pesos, n)

    bloques = []
    for perfil in nombres:
        idx = np.where(perfiles == perfil)[0]
        m = len(idx)
        if m == 0:
            continue
        p = PERFILES_CLIENTES[perfil]
        b = pd.DataFrame(index=idx)
        b["SEGMENTO_SIMULADO"] = perfil

        # --- Demografía / captación -----------------------------------------
        b["GENERO"] = _choice(rng, ["M", "F"], [0.78, 0.22] if p["prod"][0] > 0.5 else [0.6, 0.4], m)
        b["EDAD"] = np.clip(rng.normal(*p["edad"], m), 18, 85).round()
        b["PROVINCIA"] = _choice(rng, PROV_NOMBRES, PROV_PESOS, m)
        b["CANAL_ADQUISICION"] = _choice(
            rng, ["SEO", "Afiliados", "Retail", "Paid Social", "TV / Patrocinio"],
            [0.2, 0.3, 0.2, 0.2, 0.1] if perfil != "Nuevo registro" else [0.15, 0.25, 0.1, 0.4, 0.1], m)
        b["DISPOSITIVO_PRINCIPAL"] = _choice(
            rng, ["App móvil", "Web móvil", "Desktop"],
            [0.65, 0.25, 0.10] if p["edad"][0] < 40 else [0.35, 0.30, 0.35], m)
        # Los captados en local físico suelen jugar también en retail
        prob_multi = np.where(b["CANAL_ADQUISICION"] == "Retail", 0.7, 0.2)
        b["MULTICANAL"] = (rng.random(m) < prob_multi).astype(int)

        # --- Preferencias de juego --------------------------------------------
        b["PRODUCTO_FAVORITO"] = _choice(rng, ["Sports", "Casino", "Poker"], p["prod"], m)
        dep_fav = _choice(rng, ["Fútbol", "Baloncesto", "Tenis", "eSports", "Fórmula 1"], [0.68, 0.12, 0.1, 0.06, 0.04], m)
        b["DEPORTE_FAVORITO"] = np.where(b["PRODUCTO_FAVORITO"] == "Sports", dep_fav,
                                         np.where(rng.random(m) < 0.3, dep_fav, None))
        b["FRANJA_HORARIA_FAVORITA"] = _choice(
            rng, ["Mañana", "Tarde", "Noche", "Madrugada"],
            [0.1, 0.35, 0.45, 0.10] if perfil not in ("Jugador de slots", "Casino live alto valor") else [0.15, 0.25, 0.35, 0.25], m)
        b["METODO_DEPOSITO"] = _choice(
            rng, ["Tarjeta", "Bizum", "PayPal", "Transferencia", "Efectivo en local"],
            [0.45, 0.25, 0.15, 0.05, 0.10] if p["dep"] < 1000 else [0.35, 0.05, 0.10, 0.45, 0.05], m)
        b["PCT_APUESTAS_LIVE"] = np.clip(rng.normal(p["live"], 0.12, m), 0, 1).round(3)

        # --- Antigüedad, recencia y actividad ---------------------------------
        b["DIAS_DESDE_REGISTRO"] = np.clip(_lognormal(rng, p["ant"], 0.5, m), 1, 4500).round()
        rec = np.clip(_lognormal(rng, p["rec"], 0.8, m), 0, 720).round()
        b["DIAS_DESDE_ULTIMO_ACCESO"] = np.minimum(rec, b["DIAS_DESDE_REGISTRO"])
        b["DIAS_DESDE_ULTIMA_APUESTA"] = np.minimum(
            b["DIAS_DESDE_ULTIMO_ACCESO"] + rng.poisson(2 if p["acc"] > 3 else 25, m), b["DIAS_DESDE_REGISTRO"])
        b["DIAS_DESDE_ULTIMO_DEPOSITO"] = np.minimum(
            b["DIAS_DESDE_ULTIMA_APUESTA"] + rng.poisson(4 if p["dep"] > 100 else 15, m), b["DIAS_DESDE_REGISTRO"])
        b["NUM_ACCESOS_30D"] = rng.poisson(p["acc"], m)
        b["NUM_ACCESOS_90D"] = b["NUM_ACCESOS_30D"] + rng.poisson(p["acc"] * 2, m)

        # --- Dinero -----------------------------------------------------------
        dep30 = _lognormal(rng, p["dep"], 0.9, m) * (b["NUM_ACCESOS_30D"] > 0)
        b["TOTAL_DEPOSITADO_30D"] = dep30.round(2)
        b["NUM_DEPOSITOS_30D"] = np.where(dep30 > 0, np.maximum(1, rng.poisson(1 + dep30 / 80)), 0)
        meses = np.maximum(b["DIAS_DESDE_REGISTRO"] / 30, 1)
        b["TOTAL_DEPOSITADO"] = (dep30 + _lognormal(rng, max(p["dep"], 20), 0.8, m) * meses * 0.6).round(2)
        b["TOTAL_RETIRADO"] = (b["TOTAL_DEPOSITADO"] * np.clip(rng.beta(2, 5, m), 0, 0.95)).round(2)
        b["NUM_RETIROS_30D"] = rng.poisson(np.where(dep30 > 200, 1.5, 0.2))

        turnover_total = b["TOTAL_DEPOSITADO_30D"] * rng.uniform(2.5, 6, m)
        pref = b["PRODUCTO_FAVORITO"].to_numpy()
        w_sp = np.where(pref == "Sports", 0.8, np.where(pref == "Casino", 0.1, 0.1))
        w_ca = np.where(pref == "Casino", 0.85, np.where(pref == "Sports", 0.15, 0.15))
        if perfil == "Multiproducto":
            w_sp, w_ca = np.full(m, 0.45), np.full(m, 0.45)
        w_po = np.clip(1 - w_sp - w_ca, 0, 1)
        b["TURNOVER_SPORTS"] = (turnover_total * w_sp).round(2)
        b["TURNOVER_CASINO"] = (turnover_total * w_ca).round(2)
        b["TURNOVER_POKER"] = (turnover_total * w_po).round(2)
        # NGR = lo que gana la casa; puede ser negativo (cliente ganador)
        margen_sp = rng.normal(0.08, 0.15, m)
        margen_ca = rng.normal(0.04, 0.05, m)
        b["NGR_SPORTS"] = (b["TURNOVER_SPORTS"] * margen_sp).round(2)
        b["NGR_CASINO"] = (b["TURNOVER_CASINO"] * margen_ca).round(2)
        b["NGR_POKER"] = (b["TURNOVER_POKER"] * 0.05).round(2)
        b["NGR_30D"] = (b["NGR_SPORTS"] + b["NGR_CASINO"] + b["NGR_POKER"]).round(2)
        b["BALANCE"] = (_lognormal(rng, 15 + p["dep"] * 0.15, 1.1, m) * (rng.random(m) < 0.8)).round(2)
        b["NUM_APUESTAS_SPORTS_30D"] = rng.poisson(b["NUM_ACCESOS_30D"] * w_sp * 3)
        b["NUM_SESIONES_CASINO_30D"] = rng.poisson(b["NUM_ACCESOS_30D"] * w_ca * 1.5)
        n_ops = b["NUM_APUESTAS_SPORTS_30D"] + b["NUM_SESIONES_CASINO_30D"]
        b["TICKET_MEDIO"] = np.where(n_ops > 0, turnover_total / np.maximum(n_ops, 1), 0).round(2)

        # --- CRM --------------------------------------------------------------
        b["CONTACTABLE_EMAIL"] = (rng.random(m) < 0.8).astype(int)
        b["CONTACTABLE_SMS"] = (rng.random(m) < 0.55).astype(int)
        b["USA_BONOS"] = (rng.random(m) < (0.7 if perfil in ("Apostador deportivo casual", "Nuevo registro", "Jugador de slots") else 0.35)).astype(int)
        b["OPEN_RATE_EMAIL"] = np.where(b["CONTACTABLE_EMAIL"] == 1,
                                        np.clip(rng.beta(2, 6, m) + (0.15 if p["acc"] > 10 else 0), 0, 1).round(3),
                                        np.nan)
        bloques.append(b)

    df = pd.concat(bloques).sort_index()

    # Segmentación monetaria derivada (como en la ficha de cliente real)
    df["SEGMENTACION_MONETARIA"] = pd.cut(
        df["TOTAL_DEPOSITADO_30D"], bins=[-np.inf, 0, 50, 300, 1500, np.inf],
        labels=["Sin actividad", "Bajo", "Medio", "Alto", "VIP"]).astype(str)

    # Ruido realista: nulos aleatorios
    df = _inyectar_nulos(rng, df, {"EDAD": 0.02, "PROVINCIA": 0.01, "METODO_DEPOSITO": 0.03,
                                   "BALANCE": 0.02, "PCT_APUESTAS_LIVE": 0.04})

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # desordenar perfiles
    df.insert(0, "PLAYER_ID", np.arange(100001, 100001 + n))         # IDs sin patrón
    return df


# ============================================================================
# LOCALES (establecimientos con máquinas de juego)
# ============================================================================
PERFILES_LOCALES = {
    # tipo: hostelería / salón / bingo; zona; m2; máquinas B; recaudación media por máquina/mes
    "Bar de barrio": dict(peso=0.34, tipo="Hostelería", zona=("Barrio residencial", 0.8), m2=90, maq=2, rec_maq=900, horas=85, visitas=2200, antig=18),
    "Hostelería rural": dict(peso=0.12, tipo="Hostelería", zona=("Rural", 0.9), m2=110, maq=1.5, rec_maq=650, horas=70, visitas=900, antig=25),
    "Hostelería turística": dict(peso=0.08, tipo="Hostelería", zona=("Costa turística", 0.9), m2=160, maq=2.5, rec_maq=1100, horas=95, visitas=4500, antig=12),
    "Salón urbano": dict(peso=0.17, tipo="Salón de juego", zona=("Barrio residencial", 0.6), m2=260, maq=18, rec_maq=1700, horas=100, visitas=5200, antig=9),
    "Salón de paso": dict(peso=0.09, tipo="Salón de juego", zona=("Polígono / Periferia", 0.8), m2=380, maq=24, rec_maq=1500, horas=110, visitas=4800, antig=7),
    "Salón premium": dict(peso=0.06, tipo="Salón de juego", zona=("Centro urbano", 0.85), m2=520, maq=35, rec_maq=2600, horas=120, visitas=9500, antig=5),
    "Bingo tradicional": dict(peso=0.09, tipo="Bingo", zona=("Centro urbano", 0.5), m2=900, maq=30, rec_maq=1300, horas=80, visitas=7000, antig=28),
    "Bingo moderno": dict(peso=0.05, tipo="Bingo", zona=("Polígono / Periferia", 0.6), m2=1300, maq=55, rec_maq=1900, horas=105, visitas=11000, antig=6),
}
ZONAS = ["Centro urbano", "Barrio residencial", "Polígono / Periferia", "Costa turística", "Rural"]


def generar_locales(n: int, seed: int = 43) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    nombres = list(PERFILES_LOCALES)
    pesos = [PERFILES_LOCALES[p]["peso"] for p in nombres]
    perfiles = _choice(rng, nombres, pesos, n)

    bloques = []
    for perfil in nombres:
        idx = np.where(perfiles == perfil)[0]
        m = len(idx)
        if m == 0:
            continue
        p = PERFILES_LOCALES[perfil]
        es_host = p["tipo"] == "Hostelería"
        b = pd.DataFrame(index=idx)
        b["SEGMENTO_SIMULADO"] = perfil
        b["TIPO_LOCAL"] = p["tipo"]

        # --- Ubicación ---------------------------------------------------------
        zona_pref, prob_pref = p["zona"]
        resto = [z for z in ZONAS if z != zona_pref]
        b["TIPO_ZONA"] = _choice(rng, [zona_pref] + resto, [prob_pref] + [(1 - prob_pref) / 4] * 4, m)
        if perfil == "Hostelería turística":
            costa = [k for k, v in PROVINCIAS.items() if v[2]]
            pw = np.array([PROVINCIAS[k][1] for k in costa])
            b["PROVINCIA"] = _choice(rng, costa, pw, m)
        else:
            b["PROVINCIA"] = _choice(rng, PROV_NOMBRES, PROV_PESOS, m)
        b["COMUNIDAD_AUTONOMA"] = b["PROVINCIA"].map(lambda x: PROVINCIAS[x][0])
        b["NIVEL_SOCIOECONOMICO_ZONA"] = _choice(
            rng, ["Bajo", "Medio", "Alto"],
            [0.15, 0.45, 0.40] if perfil == "Salón premium" else
            [0.40, 0.45, 0.15] if perfil in ("Bar de barrio", "Salón urbano", "Salón de paso") else [0.3, 0.5, 0.2], m)
        pob_base = {"Centro urbano": 38000, "Barrio residencial": 22000, "Polígono / Periferia": 9000,
                    "Costa turística": 15000, "Rural": 1800}
        b["POBLACION_ZONA_1KM"] = (b["TIPO_ZONA"].map(pob_base) * rng.lognormal(0, 0.35, m)).round()
        b["COMPETIDORES_500M"] = rng.poisson(np.where(b["TIPO_ZONA"].isin(["Centro urbano", "Barrio residencial"]), 4, 1.2))

        # --- Estructura del local ----------------------------------------------
        b["SUPERFICIE_M2"] = np.clip(_lognormal(rng, p["m2"], 0.3, m), 30, 3000).round()
        b["AFORO"] = (b["SUPERFICIE_M2"] * rng.uniform(0.6, 1.1, m)).round()
        b["ANTIGUEDAD_ANIOS"] = np.clip(rng.gamma(2, p["antig"] / 2, m), 0, 60).round(1)
        if es_host:
            b["NUM_MAQUINAS_B"] = np.clip(rng.poisson(p["maq"], m), 1, 3)
        else:
            b["NUM_MAQUINAS_B"] = np.clip(rng.normal(p["maq"], p["maq"] * 0.25, m), 4, 120).round()
        b["NUM_EMPLEADOS"] = np.maximum(1, rng.poisson(1.5 if es_host else b["NUM_MAQUINAS_B"] / 6))
        b["HORAS_APERTURA_SEMANA"] = np.clip(rng.normal(p["horas"], 10, m), 30, 168).round()
        b["HORARIO"] = np.where(b["HORAS_APERTURA_SEMANA"] >= 115, "Extendido",
                                np.where(b["HORAS_APERTURA_SEMANA"] >= 80, "Estándar", "Reducido"))

        # --- Servicios (booleanos) ---------------------------------------------
        b["TIENE_APUESTAS_DEPORTIVAS"] = (rng.random(m) < (0.15 if es_host else 0.8)).astype(int)
        b["NUM_TERMINALES_APUESTAS"] = np.where(b["TIENE_APUESTAS_DEPORTIVAS"] == 1,
                                                np.maximum(1, rng.poisson(1 if es_host else 4, m)), 0)
        b["TIENE_RESTAURACION"] = (rng.random(m) < (0.97 if es_host else 0.55 if p["tipo"] == "Bingo" else 0.35)).astype(int)
        b["TIENE_TERRAZA"] = (rng.random(m) < (0.6 if es_host else 0.1)).astype(int)
        b["TIENE_PARKING"] = (rng.random(m) < np.where(b["TIPO_ZONA"].isin(["Polígono / Periferia", "Rural"]), 0.8, 0.15)).astype(int)
        b["PROGRAMA_FIDELIZACION"] = (rng.random(m) < (0.1 if es_host else 0.75)).astype(int)
        b["PROVEEDOR_MAQUINAS"] = _choice(rng, [f"Proveedor {c}" for c in "ABCDE"],
                                          [0.35, 0.25, 0.2, 0.15, 0.05] if es_host else [0.15, 0.2, 0.3, 0.25, 0.1], m)

        # --- Negocio ---------------------------------------------------------
        socio = b["NIVEL_SOCIOECONOMICO_ZONA"].map({"Bajo": 0.9, "Medio": 1.0, "Alto": 1.15})
        rec_maq = _lognormal(rng, p["rec_maq"], 0.35, m) * socio
        b["RECAUDACION_MEDIA_MENSUAL"] = (rec_maq * b["NUM_MAQUINAS_B"]).round(2)
        b["VARIACION_RECAUDACION_YOY"] = np.clip(rng.normal(0.02 if not es_host else -0.03, 0.12, m), -0.6, 0.8).round(3)
        b["RECAUDACION_ULTIMO_MES"] = (b["RECAUDACION_MEDIA_MENSUAL"] * (1 + b["VARIACION_RECAUDACION_YOY"])
                                       * rng.normal(1, 0.08, m)).round(2)
        b["RECAUDACION_POR_MAQUINA"] = (b["RECAUDACION_MEDIA_MENSUAL"] / b["NUM_MAQUINAS_B"]).round(2)
        b["VISITAS_MES"] = np.clip(_lognormal(rng, p["visitas"], 0.35, m), 50, 60000).round()
        b["TICKET_MEDIO"] = (b["RECAUDACION_MEDIA_MENSUAL"] / b["VISITAS_MES"] * rng.uniform(2.5, 4, m)).round(2)
        b["PCT_OCUPACION_MEDIA"] = np.clip(rng.beta(4, 5, m) + (0.1 if perfil in ("Salón premium", "Bingo moderno") else 0), 0, 1).round(3)

        # --- Operaciones ---------------------------------------------------
        b["INCIDENCIAS_TECNICAS_90D"] = rng.poisson(b["NUM_MAQUINAS_B"] * (0.12 + b["ANTIGUEDAD_ANIOS"] / 400))
        b["DIAS_DESDE_ULTIMO_RECANJE"] = np.clip(rng.gamma(2, 180 if es_host else 120, m), 0, 2500).round()
        bloques.append(b)

    df = pd.concat(bloques).sort_index()
    df = _inyectar_nulos(rng, df, {"POBLACION_ZONA_1KM": 0.03, "ANTIGUEDAD_ANIOS": 0.02,
                                   "PCT_OCUPACION_MEDIA": 0.05, "PROVEEDOR_MAQUINAS": 0.02,
                                   "VARIACION_RECAUDACION_YOY": 0.03})
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "LOCAL_ID", np.arange(5001, 5001 + n))
    return df


# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Genera los datasets sintéticos en Excel")
    parser.add_argument("--n-clientes", type=int, default=10_000)
    parser.add_argument("--n-locales", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    clientes = generar_clientes(args.n_clientes, args.seed)
    clientes.to_excel(DATA_DIR / "clientes.xlsx", index=False, sheet_name="clientes")
    print(f"✓ data/clientes.xlsx -> {clientes.shape[0]:,} filas x {clientes.shape[1]} columnas")
    print(clientes["SEGMENTO_SIMULADO"].value_counts().to_string(), "\n")

    locales = generar_locales(args.n_locales, args.seed + 1)
    locales.to_excel(DATA_DIR / "locales.xlsx", index=False, sheet_name="locales")
    print(f"✓ data/locales.xlsx  -> {locales.shape[0]:,} filas x {locales.shape[1]} columnas")
    print(locales["SEGMENTO_SIMULADO"].value_counts().to_string())


if __name__ == "__main__":
    main()
