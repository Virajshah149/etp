# K-Means Methodology (Ghost Hotspot Detection)

Here is the explanation you requested for the K-Means methodology used in the Eco-Transit Pulse project. You can use this for your report or presentation.

## Clustering Features
The K-Means algorithm groups bus stops based on three primary features:
- **Latitude & Longitude**: To capture geographic positioning and spatial proximity.
- **Average Passenger Demand per Hour**: To understand the ridership load at each location.

## Algorithm Details
- **Algorithm Used**: K-Means Clustering
- **Number of Clusters (k)**: 5
- **Preprocessing**: Data was normalized using `StandardScaler` to ensure geographic coordinates and passenger demand are weighed evenly during distance calculation.

## Ghost Hotspot Criteria
A "Ghost Hotspot" is defined as a bus stop that experiences high passenger demand but suffers from low transit coverage. Specifically, it must meet both of these conditions:
1. **High Demand**: The average passenger demand is greater than or equal to the 75th percentile of all stops in the network.
2. **Low Coverage**: The stop is served by only 1 bus route.

## Why k=5?
The optimal number of clusters was chosen using the **Elbow Method**. The sum of squared distances (inertia) was computed for k values ranging from 2 to 10. The resulting plot showed a distinct "elbow" (the point of diminishing returns) at k=5, indicating that adding more clusters beyond 5 did not significantly improve the grouping.
