FROM influxdb:latest

WORKDIR /app
COPY entrypoint.sh ./
RUN chmod u+x entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

#EXPOSE 8086 8125/udp 8092/udp 8094
# Admin server
EXPOSE 8083

# HTTP API
EXPOSE 8086

# HTTPS API
EXPOSE 8084

# Raft port
EXPOSE 8090

# Protobuf port
EXPOSE 8099
