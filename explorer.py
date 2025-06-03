import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np

    return np, pd


@app.cell
def _(pd):
    df = pd.read_parquet("data\wildfire_integrated.parquet")
    return (df,)


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df, np, pd):
    # Step 1: Extract unique cluster centers
    clusters = (
        df[["fire_cluster_id", "cluster_center_lat", "cluster_center_lon"]]
        .drop_duplicates()
        .set_index("fire_cluster_id")
    )

    def generate_cluster_grids(clusters):
        """
        For each cluster center (latitude, longitude), generate a 1km x 1km grid
        covering a 10km radius around the point (i.e., a 20km x 20km square).
        Returns a dict: {cluster_id: DataFrame of cell-center lat/lon}.
        """
        grids = {}
        for cid, row in clusters.iterrows():
            lat0 = row["cluster_center_lat"]
            lon0 = row["cluster_center_lon"]
            # Approximate degrees per km
            deg_per_km_lat = 1 / 111.0  # ~0.009009 deg latitude per km
            deg_per_km_lon = 1 / (
                111.0 * np.cos(np.deg2rad(lat0))
            )  # adjust for latitude

            # Half side length in degrees for 10km
            dlat_half = 10 * deg_per_km_lat
            dlon_half = 10 * deg_per_km_lon

            # Bounding box in lat/lon
            lat_min = lat0 - dlat_half
            lat_max = lat0 + dlat_half
            lon_min = lon0 - dlon_half
            lon_max = lon0 + dlon_half

            # Step sizes for 1km grid
            lat_step = deg_per_km_lat
            lon_step = deg_per_km_lon

            # Generate grid of cell centers
            lats = np.arange(lat_min + lat_step / 2, lat_max, lat_step)
            lons = np.arange(lon_min + lon_step / 2, lon_max, lon_step)

            mesh_lon, mesh_lat = np.meshgrid(lons, lats)
            grid_df = pd.DataFrame(
                {"lat_center": mesh_lat.ravel(), "lon_center": mesh_lon.ravel()}
            )
            grids[cid] = grid_df
        return grids

    # Generate grids for all clusters
    cluster_grids = generate_cluster_grids(clusters)

    # Example: inspect the first few cells for cluster_id = 1
    cluster_grids[1].head()
    return cluster_grids, clusters


@app.cell
def _(cluster_grids, clusters, df, np, pd):
    # Assume `df` is your full DataFrame, and `cluster_grids` from the previous step.

    # CA parameters (tune as needed)
    p0 = 0.1  # base spread probability
    alpha = 0.02  # wind-scaling factor
    max_steps = 15  # maximum number of CA iterations

    # Precompute neighbor offsets and their bearings (radians from East, CCW)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    bearings = [
        135 * np.pi / 180,
        180 * np.pi / 180,
        225 * np.pi / 180,
        90 * np.pi / 180,
        0 * np.pi / 180,
        45 * np.pi / 180,
        270 * np.pi / 180,
        315 * np.pi / 180,
    ]

    def simulate_cluster_fire(cluster_id):
        # 1) Extract grid and cluster center for this ID
        grid_df = cluster_grids[cluster_id]
        lat0 = clusters.loc[cluster_id, "cluster_center_lat"]
        lon0 = clusters.loc[cluster_id, "cluster_center_lon"]

        # 2) Get uniform wind speed & direction (radians) from any one detection of this cluster
        sample = df[df["fire_cluster_id"] == cluster_id].iloc[0]
        U_cluster = sample["current_wind_speed_10m"]
        theta_wind = np.deg2rad(sample["current_wind_direction_10m"])

        # 3) Build 2D grid arrays
        lats = np.sort(grid_df["lat_center"].unique())
        lons = np.sort(grid_df["lon_center"].unique())
        nlat, nlon = len(lats), len(lons)

        # Map each cell into a 2D index
        lat_to_i = {lat: i for i, lat in enumerate(lats)}
        lon_to_j = {lon: j for j, lon in enumerate(lons)}

        # Initialize cell states: 0=unburned, 1=burning, 2=burned
        grid = np.zeros((nlat, nlon), dtype=np.uint8)

        # 4) Find center cell index
        i0 = np.argmin(np.abs(lats - lat0))
        j0 = np.argmin(np.abs(lons - lon0))
        grid[i0, j0] = 1  # ignite center

        # 5) Uniform wind arrays
        wind_u = np.full((nlat, nlon), U_cluster, dtype=float)
        wind_theta = np.full((nlat, nlon), theta_wind, dtype=float)

        def step_CA(grid, wind_u, wind_theta):
            new_grid = grid.copy()
            for i in range(nlat):
                for j in range(nlon):
                    if grid[i, j] != 1:
                        continue
                    U = wind_u[i, j]
                    theta_w = wind_theta[i, j]
                    for (dx, dy), theta_i in zip(offsets, bearings):
                        ni, nj = i + dx, j + dy
                        if not (0 <= ni < nlat and 0 <= nj < nlon):
                            continue
                        if grid[ni, nj] != 0:
                            continue
                        phi = np.cos(theta_i - theta_w)
                        P = p0 * max(0, 1 + alpha * U * phi)
                        if np.random.rand() < P:
                            new_grid[ni, nj] = 1
                    new_grid[i, j] = 2
            return new_grid

        # 6) Run the CA
        for step in range(max_steps):
            if not (grid == 1).any():
                break
            grid = step_CA(grid, wind_u, wind_theta)

        return grid, lats, lons

    # Example: simulate for cluster 1
    burned_mask, lat_edges, lon_edges = simulate_cluster_fire(cluster_id=1)

    # Convert final mask to DataFrame for inspection
    lat_idx, lon_idx = np.where(burned_mask == 2)
    result_cells = pd.DataFrame(
        {
            "lat_center": lat_edges[lat_idx],
            "lon_center": lon_edges[lon_idx],
            "state": "burned",
        }
    )
    result_cells.head()

    return (result_cells,)


@app.cell
def _(result_cells):
    result_cells
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
