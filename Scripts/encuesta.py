import pandas as pd
import numpy as np

def pivot_fechas(df):
    df_p = df.copy()
    col_cliente = next(
        (c for c in ["llave_cliente", "Cod Cliente", "COD CLIENTE"] if c in df_p.columns),
        None
    )
    if col_cliente is None:
        raise ValueError("No se encontró columna de cliente")
    columnas_fijas = [col_cliente]
    for c in ["Nombre Region", "Nombre Cliente HML"]:
        if c in df_p.columns:
            columnas_fijas.append(c)
    columnas_periodos = [c for c in df_p.columns if c not in columnas_fijas]
    df_long = df_p.melt(
        id_vars=columnas_fijas,
        value_vars=columnas_periodos,
        var_name="Fecha",
        value_name="Valor"
    )
    df_long["Fecha"] = pd.to_datetime(df_long["Fecha"], format="%Y-%m", errors="coerce")
    df_long["Fecha"] = df_long["Fecha"].dt.to_period("M").dt.to_timestamp()
    df_long = df_long[df_long[col_cliente].notna() & df_long["Valor"].notna()]
    cols_grp = [col_cliente, "Fecha"]
    df_long = (
        df_long
        .groupby(cols_grp, as_index=False)
        .agg(Valor=("Valor", "sum"))
    )
    return df_long

def SO_CambioRUC(df_Ventas, df_CambioRUC):
    df_Ventas = df_Ventas.copy()
    df_CambioRUC = df_CambioRUC.copy()
    map_ruc = dict(
        zip(df_CambioRUC["Cod Cliente Antiguo"],
            df_CambioRUC["Cod Cliente"])
    )
    df_Ventas["llave_cliente"] = (
        df_Ventas["llave_cliente"]
        .replace(map_ruc)
    )
    df_Ventas_final = (
        df_Ventas
        .groupby(["llave_cliente", "Fecha"], as_index=False)
        .agg(Valor=("Valor", "sum"))
    )
    return df_Ventas_final

def unir_materiales(series):
    materiales = set()
    for val in series:
        if pd.isna(val):
            continue
        val = str(val).strip()
        if val == "":
            continue
        for m in val.split(","):
            m = m.strip()
            if m:
                materiales.add(m)
    materiales_ordenados = sorted(materiales)
    return ",".join(materiales_ordenados), len(materiales_ordenados)

def procesar_flujo_embajadores(archivo):
    dfs = archivo.copy()

    df_Ventas = pivot_fechas(dfs['df_Ventas'].fillna(0))
    df_Ventas = SO_CambioRUC(df_Ventas, dfs['df_CambioRUC'])

    df_Ventas['Fecha_LY'] = df_Ventas['Fecha'] - pd.DateOffset(years=1)
    
    df_aux = df_Ventas[['llave_cliente', 'Fecha', 'Valor']].rename(
        columns={
            'Fecha': 'Fecha_LY_match',
            'Valor': 'Valor_LY'
        }
    )
    
    df_resultado = df_Ventas.merge(
        df_aux,
        left_on=['llave_cliente', 'Fecha_LY'],
        right_on=['llave_cliente', 'Fecha_LY_match'],
        how='left'
    ).drop(columns=['Fecha_LY', 'Fecha_LY_match'])

    df_CuotasC = dfs['df_Cuotas']
    df_Cuotas = pivot_fechas(dfs['df_Cuotas'])

    cl = df_resultado.merge(
        df_Cuotas,
        left_on=["llave_cliente", "Fecha"],
        right_on=["COD CLIENTE", "Fecha"],
        how="outer"
    ).dropna(subset=["llave_cliente"]).drop(columns="COD CLIENTE")

    cl["Fecha"] = pd.to_datetime(cl["Fecha"]).dt.to_period("M").dt.to_timestamp()

    df_Enc = dfs['df_Encuesta'].copy()
    df_Enc["Fecha"] = pd.to_datetime(df_Enc["Created on"], errors="coerce").dt.to_period("M").dt.to_timestamp()

    df_Enc = df_Enc[df_Enc["Cliente pertenece a plan de fidelización?"] == "SI"]

    map_bool = {'SI': 1, 'Sí': 1, 'YES': 1, 'Yes': 1, True: 1, 'NO': 0, 'No': 0, False: 0}
    bool_cols = ['Máquina contaminada?', 'Nuestra máquina está en primera posición?', 'Maquina de la Competencia']

    for col in bool_cols:
        df_Enc[col] = df_Enc[col].map(map_bool).fillna(0).astype(int)

    MAP_LLENADO = {"≥ 80%": 1, "50% - 80%": 2, "0 - 50%": 3, "0": 4}
    df_Enc["llenado_cod"] = df_Enc["Llenado de la máquina"].astype(str).str.strip().map(MAP_LLENADO).fillna(4).astype(int)

    resumen = df_Enc.groupby(["Cod Cliente", "Distribuidor", "Fecha"], as_index=False).agg(
        **{col: (col, "max") for col in bool_cols},
        llenado_final=("llenado_cod", "min"),
        materiales_info=("¿Cumple foto del éxito?", unir_materiales),
        productos_info=("Disponibilidad de productos", unir_materiales)
    )

    resumen[["materiales_str", "n_materiales"]] = pd.DataFrame(resumen["materiales_info"].tolist(), index=resumen.index)
    resumen[["productos_str", "n_productos"]] = pd.DataFrame(resumen["productos_info"].tolist(), index=resumen.index)

    dummies = resumen["materiales_str"].str.get_dummies(sep=",")
    
    dummies_productos = resumen["productos_str"].str.get_dummies(sep=",")

    resumen_final = pd.concat([resumen.drop(columns=["materiales_info", "materiales_str", "productos_info", "productos_str"]), dummies, dummies_productos], axis=1)

    meses_sin_productos = pd.to_datetime(["2026-01-01","2026-02-01"])
    mask_skip = resumen_final["Fecha"].isin(meses_sin_productos)
    
    cond_productos = (resumen_final["n_productos"] >= 5) | mask_skip
    
    resumen_final["Procede"] = (
        (resumen_final['Nuestra máquina está en primera posición?'] == 1) &
        (resumen_final['llenado_final'].isin([1,2])) &
        (resumen_final['Máquina contaminada?'] == 0) &
        (resumen_final["n_materiales"] >= 3) &
        cond_productos
    ).astype(int)

    resumen_final["Encuesta"] = "SI"

    encabezado_cuotas=[
        "COD CLIENTE",
        "REGIÓN",
        "DT",
        "NOMBRE CLIENTE",
        "CLÚSTER",
        "CANAL",
        "STATUS"
    ]

    df_CF = df_CuotasC[encabezado_cuotas][df_CuotasC["CANAL"]=="HORIZONTAL"]
    resultado = []

    for mes in resumen_final['Fecha'].unique():

        enc_mes = resumen_final[resumen_final['Fecha'] == mes]

        merge_mes = df_CF.merge(
            enc_mes,
            left_on='COD CLIENTE',
            right_on='Cod Cliente',
            how='left'
        )

        merge_mes['Mes'] = mes

        resultado.append(merge_mes)

    df_consolidado = pd.concat(resultado, ignore_index=True)

    df_consolidado["Mes"] = pd.to_datetime(df_consolidado["Mes"])
    cl["Fecha"] = pd.to_datetime(cl["Fecha"])
    
    rp = df_consolidado.merge(cl, left_on=["COD CLIENTE", "Mes"], right_on=["llave_cliente", "Fecha"], how="left").fillna(0)
    rp = rp.drop(columns=["llave_cliente", "Cod Cliente", "Distribuidor", "Fecha_x", "Fecha_y"])

    MAP_LLENADO_INV = {1: "≥ 80%", 2: "50% - 80%", 3: "0 - 50%", 4: "0"}
    rp["Llenado de la máquina"] = rp["llenado_final"].astype(int).map(MAP_LLENADO_INV)

    for col in bool_cols:
        rp[col] = rp[col].replace({0: "NO", 1: "SI"})

    rp['Valor_x'] = pd.to_numeric(rp['Valor_x'], errors='coerce')
    rp['Valor_y'] = pd.to_numeric(rp['Valor_y'], errors='coerce')
    rp['Valor_LY'] = pd.to_numeric(rp['Valor_LY'], errors='coerce')
        
    rp['Valor_LY'] = np.where((rp['Valor_y'] == 0) | (rp['Valor_y'].isna()), 0, rp['Valor_LY'])
    rp['%Avance'] = np.divide(
        rp['Valor_x'],
        rp['Valor_y'],
        out=np.zeros_like(rp['Valor_x'], dtype=float),
        where=(rp['Valor_y'] != 0)
    )
    rp['%Crecimiento'] = np.divide(
        rp['Valor_x'],
        rp['Valor_LY'],
        out=np.zeros_like(rp['Valor_x'], dtype=float),
        where=(rp['Valor_LY'] != 0)
    )

    rp['Ruptura'] = np.where(rp['Valor_x'] <= 0, "Ruptura", "Ok")

    return rp
