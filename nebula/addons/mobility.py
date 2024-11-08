import logging
import random
import math
import threading
import time
from nebula.addons.functions import print_msg_box
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nebula.core.network.communications import CommunicationsManager


class Mobility(threading.Thread):
    def __init__(self, config, cm: "CommunicationsManager"):
        threading.Thread.__init__(self, daemon=True, name="mobility_thread-" + config.participant["device_args"]["name"])
        logging.info(f"Starting mobility thread")
        self.config = config
        self.cm = cm
        self.grace_time = self.config.participant["mobility_args"]["grace_time_mobility"]
        self.period = self.config.participant["mobility_args"]["change_geo_interval"]
        self.mobility = self.config.participant["mobility_args"]["mobility"]
        self.mobility_type = self.config.participant["mobility_args"]["mobility_type"]
        self.radius_federation = float(self.config.participant["mobility_args"]["radius_federation"])
        self.scheme_mobility = self.config.participant["mobility_args"]["scheme_mobility"]
        self.round_frequency = int(self.config.participant["mobility_args"]["round_frequency"])
        # Protocol to change connections based on distance
        self.max_distance_with_direct_connections = 300  # meters
        self.max_movement_random_strategy = 100  # meters
        self.max_movement_nearest_strategy = 100  # meters
        self.max_initiate_approximation = self.max_distance_with_direct_connections * 1.2
        # Network conditions based on distance
        self.network_conditions = {100: {"bandwidth": "5Gbps", "delay": "5ms"}, 200: {"bandwidth": "2Gbps", "delay": "50ms"}, 300: {"bandwidth": "100Mbps", "delay": "200ms"}, float("inf"): {"bandwidth": "10Mbps", "delay": "1000ms"}}
        # Current network conditions of each connection {addr: {bandwidth: "5Gbps", delay: "0ms"}}
        self.current_network_conditions = {}
        # Logging box with mobility information
        mobility_msg = f"Mobility: {self.mobility}\nMobility type: {self.mobility_type}\nRadius federation: {self.radius_federation}\nScheme mobility: {self.scheme_mobility}\nEach {self.round_frequency} rounds"
        print_msg_box(msg=mobility_msg, indent=2, title="Mobility information")

    @property
    def round(self):
        return self.cm.get_round()

    def run(self):
        if not self.mobility:
            return
        time.sleep(self.grace_time)
        while True:
            self.change_geo_location()
            self.change_connections_based_on_distance()
            time.sleep(self.period)

    def change_geo_location_random_strategy(self, latitude, longitude):
        logging.info(f"📍  Changing geo location randomly")
        # radius_in_degrees = self.radius_federation / 111000
        max_radius_in_degrees = self.max_movement_random_strategy / 111000
        radius = random.uniform(0, max_radius_in_degrees)
        angle = random.uniform(0, 2 * math.pi)
        latitude += radius * math.cos(angle)
        longitude += radius * math.sin(angle)
        self.set_geo_location(latitude, longitude)

    def change_geo_location_nearest_neighbor_strategy(self, distance, latitude, longitude, neighbor_latitude, neighbor_longitude):
        logging.info(f"📍  Changing geo location towards the nearest neighbor")
        scale_factor = min(1, self.max_movement_nearest_strategy / distance)
        # Calcular el ángulo hacia el vecino
        angle = math.atan2(neighbor_longitude - longitude, neighbor_latitude - latitude)
        # Conversión de movimiento máximo a grados
        max_lat_change = self.max_movement_nearest_strategy / 111000  # Cambio en grados para latitud
        max_lon_change = self.max_movement_nearest_strategy / (111000 * math.cos(math.radians(latitude)))  # Cambio en grados para longitud
        # Aplicar escala y dirección
        delta_lat = max_lat_change * math.cos(angle) * scale_factor
        delta_lon = max_lon_change * math.sin(angle) * scale_factor
        # Actualizar latitud y longitud
        new_latitude = latitude + delta_lat
        new_longitude = longitude + delta_lon
        self.set_geo_location(new_latitude, new_longitude)

    def set_geo_location(self, latitude, longitude):
        if latitude < -90 or latitude > 90 or longitude < -180 or longitude > 180:
            # If the new location is out of bounds, we keep the old location
            latitude = self.config.participant["mobility_args"]["latitude"]
            longitude = self.config.participant["mobility_args"]["longitude"]

        self.config.participant["mobility_args"]["latitude"] = latitude
        self.config.participant["mobility_args"]["longitude"] = longitude
        logging.info(f"📍  New geo location: {latitude}, {longitude}")

    def change_geo_location(self):
        if self.mobility and (self.mobility_type == "topology" or self.mobility_type == "both"):
            random.seed(time.time() + self.config.participant["device_args"]["idx"])
            latitude = float(self.config.participant["mobility_args"]["latitude"])
            longitude = float(self.config.participant["mobility_args"]["longitude"])

            direct_connections = self.cm.get_direct_connections()
            undirect_connection = self.cm.get_undirect_connections()
            if len(undirect_connection) > len(direct_connections):
                logging.info(f"📍  Undirect Connections is higher than Direct Connections")
                # Get neighbor closer to me
                selected_neighbor = self.cm.get_nearest_connections(top=1)
                logging.info(f"📍  Selected neighbor: {selected_neighbor}")
                try:
                    neighbor_latitude, neighbor_longitude = selected_neighbor.get_geolocation()
                    distance = selected_neighbor.get_neighbor_distance()
                    if distance > self.max_initiate_approximation:
                        # If the distance is too big, we move towards the neighbor
                        self.change_geo_location_nearest_neighbor_strategy(distance, latitude, longitude, neighbor_latitude, neighbor_longitude)
                    else:
                        self.change_geo_location_random_strategy(latitude, longitude)
                except Exception as e:
                    logging.info(f"📍  Neighbor location/distance not found for {selected_neighbor.get_addr()}: {e}")
                    self.change_geo_location_random_strategy(latitude, longitude)
            else:
                self.change_geo_location_random_strategy(latitude, longitude)
        else:
            logging.error(f"📍  Mobility type {self.mobility_type} not implemented")
            return

    def change_connections_based_on_distance(self):
        if self.mobility and (self.mobility_type == "topology" or self.mobility_type == "both"):
            try:
                # logging.info(f"📍  Checking connections based on distance")
                connections_topology = self.cm.get_addrs_current_connections()
                # logging.info(f"📍  Connections of the topology: {connections_topology}")
                if len(connections_topology) < 1:
                    # logging.error(f"📍  Not enough connections for mobility")
                    return
                # Nodes that are too far away should be marked as undirected connections, and closer nodes should be marked as directed connections.
                for addr in connections_topology:
                    distance = self.cm.connections[addr].get_neighbor_distance()
                    if distance is None:
                        # If the distance is not found, we skip the node
                        continue
                    # logging.info(f"📍  Distance to node {addr}: {distance}")
                    if not self.cm.connections[addr].get_direct() and distance < self.max_distance_with_direct_connections:
                        logging.info(f"📍  Node {addr} is close enough [{distance}], adding to direct connections")
                        self.cm.connections[addr].set_direct(True)
                    else:
                        # 10% margin to avoid oscillations
                        if self.cm.connections[addr].get_direct() and distance > self.max_distance_with_direct_connections * 1.1:
                            logging.info(f"📍  Node {addr} is too far away [{distance}], removing from direct connections")
                            self.cm.connections[addr].set_direct(False)
                    # Adapt network conditions of the connection based on distance
                    for threshold in sorted(self.network_conditions.keys()):
                        if distance < threshold:
                            conditions = self.network_conditions[threshold]
                            break
                    # Only update the network conditions if they have changed
                    if addr not in self.current_network_conditions or self.current_network_conditions[addr] != conditions:
                        # eth1 is the interface of the container that connects to the node network - eth0 is the interface of the container that connects to the frontend/backend
                        self.cm._set_network_conditions(interface="eth1", network=addr.split(":")[0], bandwidth=conditions["bandwidth"], delay=conditions["delay"], delay_distro="10ms", delay_distribution="normal", loss="0%", duplicate="0%", corrupt="0%", reordering="0%")
                        self.current_network_conditions[addr] = conditions
            except KeyError as e:
                # Except when self.cm.connections[addr] is not found (disconnected during the process)
                logging.error(f"📍  Connection {addr} not found: {e}")
                return
            except Exception as e:
                logging.error(f"📍  Error changing connections based on distance: {e}")
                return

    async def change_connections(self):
        if self.mobility and (self.mobility_type == "topology" or self.mobility_type == "both") and self.round % self.round_frequency == 0:
            logging.info(f"📍  Changing connections")
            current_connections = self.cm.get_addrs_current_connections(only_direct=True)
            potential_connections = self.cm.get_addrs_current_connections(only_undirected=True)
            logging.info(f"📍  Current connections: {current_connections} | Potential future connections: {potential_connections}")
            if len(current_connections) < 1 or len(potential_connections) < 1:
                logging.error(f"📍  Not enough connections for mobility")
                return

            if self.scheme_mobility == "random":
                random_neighbor = random.choice(current_connections)
                random_potential_neighbor = random.choice(potential_connections)
                logging.info(f"📍  Selected node(s) to disconnect: {random_neighbor}")
                logging.info(f"📍  Selected node(s) to connect: {random_potential_neighbor}")
                self.cm.disconnect(random_neighbor, mutual_disconnection=True)
                self.cm.connect(random_potential_neighbor, direct=True)
                logging.info(f"📍  New connections: {self.get_current_connections(only_direct=True)}")
                logging.info(f"📍  Neighbors in config: {self.config.participant['network_args']['neighbors']}")
            elif self.scheme_mobility == "vic":
                # Change connections based on VIC flight formation: formation devised for military aircraft and first used during the First World War. It has three or sometimes more aircraft fly in close formation with the leader at the apex and the rest of the flight en echelon to the left and the right, the whole resembling the letter "V".
                # Change my latitude and longitude to the center of the formation, and connect to the nearest neighbors
                # Get the neighbors closer to me
                selected_neighbor = self.cm.get_nearest_connections(top=1)
                logging.info(f"📍  Selected neighbor: {selected_neighbor}" f" | Distance: {selected_neighbor.get_neighbor_distance()}")
                neighbor_latitude, neighbor_longitude = selected_neighbor.get_geolocation()
                # Calculate the center of the formation
                formation_latitude = (float(self.config.participant["mobility_args"]["latitude"]) + neighbor_latitude) / 2
                formation_longitude = (float(self.config.participant["mobility_args"]["longitude"]) + neighbor_longitude) / 2
                self.set_geo_location(formation_latitude, formation_longitude)
                # Connect to the nearest neighbors
                self.cm.connect(selected_neighbor.get_addr(), direct=True)
                logging.info(f"📍  New connections: {self.cm.get_addrs_current_connections(only_direct=True)}" f" | Neighbors in config: {self.config.participant['network_args']['neighbors']}")
            elif self.scheme_mobility == "diamond":
                # Change connections based on diamond formation
                # Change my latitude and longitude to a new location, and connect to the nearest neighbors
                # Update the locations based on the diamond formation (predefined locations)
                diamond_locations = [
                    {"latitude": self.config.participant["mobility_args"]["latitude"], "longitude": self.config.participant["mobility_args"]["longitude"]},
                    {"latitude": self.config.participant["mobility_args"]["latitude"] + 0.1, "longitude": self.config.participant["mobility_args"]["longitude"]},
                    {"latitude": self.config.participant["mobility_args"]["latitude"] - 0.1, "longitude": self.config.participant["mobility_args"]["longitude"]},
                    {"latitude": self.config.participant["mobility_args"]["latitude"], "longitude": self.config.participant["mobility_args"]["longitude"] + 0.1},
                    {"latitude": self.config.participant["mobility_args"]["latitude"], "longitude": self.config.participant["mobility_args"]["longitude"] - 0.1},
                ]

                logging.info(f"📍  Changing geo location to a new location based on diamond formation" f" | New location: {diamond_locations[self.round % len(diamond_locations)]}")
                self.set_geo_location(diamond_locations[self.round % len(diamond_locations)]["latitude"], diamond_locations[self.round % len(diamond_locations)]["longitude"])
                # Connect to the nearest neighbors
                selected_neighbor = self.cm.get_nearest_connections(top=1)
                logging.info(f"📍  Selected neighbor: {selected_neighbor}" f" | Distance: {selected_neighbor.get_neighbor_distance()}")
                self.cm.connect(selected_neighbor.get_addr(), direct=True)
                logging.info(f"📍  New connections: {self.cm.get_addrs_current_connections(only_direct=True)}" f" | Neighbors in config: {self.config.participant['network_args']['neighbors']}")
            elif self.scheme_mobility == "line":
                # Change connections based on line formation
                # Change my latitude and longitude to a new location, and connect to the nearest neighbors
                # Update the locations based on the line formation (predefined locations)
                line_locations = [
                    {"latitude": self.config.participant["mobility_args"]["latitude"], "longitude": self.config.participant["mobility_args"]["longitude"]},
                    {"latitude": self.config.participant["mobility_args"]["latitude"] + 0.1, "longitude": self.config.participant["mobility_args"]["longitude"]},
                    {"latitude": self.config.participant["mobility_args"]["latitude"] + 0.2, "longitude": self.config.participant["mobility_args"]["longitude"]},
                    {"latitude": self.config.participant["mobility_args"]["latitude"] + 0.3, "longitude": self.config.participant["mobility_args"]["longitude"]},
                    {"latitude": self.config.participant["mobility_args"]["latitude"] + 0.4, "longitude": self.config.participant["mobility_args"]["longitude"]},
                ]

                logging.info(f"📍  Changing geo location to a new location based on line formation" f" | New location: {line_locations[self.round % len(line_locations)]}")
                self.set_geo_location(line_locations[self.round % len(line_locations)]["latitude"], line_locations[self.round % len(line_locations)]["longitude"])
                # Connect to the nearest neighbors
                selected_neighbor = self.cm.get_nearest_connections(top=1)
                logging.info(f"📍  Selected neighbor: {selected_neighbor}" f" | Distance: {selected_neighbor.get_neighbor_distance()}")
                self.cm.connect(selected_neighbor.get_addr(), direct=True)
                logging.info(f"📍  New connections: {self.cm.get_addrs_current_connections(only_direct=True)}" f" | Neighbors in config: {self.config.participant['network_args']['neighbors']}")
            else:
                logging.error(f"📍  Mobility scheme {self.scheme_mobility} not implemented")
                return
