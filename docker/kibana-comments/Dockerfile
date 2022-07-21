ARG ES_VERSION=7.3.0
ARG LICENSE

FROM tatsui/kibana-plugin-builder:${ES_VERSION} as comments
ENV PLUGIN_NAME=comments
COPY --chown=node:node comments /plugin/kibana-extra/comments
RUN build${LICENSE}.sh

FROM docker.elastic.co/kibana/kibana${LICENSE}:${ES_VERSION}

COPY --from=comments /plugin/kibana-extra/comments/build/*.zip /tmp/comments.zip
RUN kibana-plugin install file:///tmp/comments.zip
RUN kibana --optimize
