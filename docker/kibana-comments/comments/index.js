import { resolve } from 'path';
import { initServer } from './server';
import serverRouteEsComment from './server/routes/esComment';
import serverRouteEsIndex from './server/routes/esIndex';


export default function (kibana) {
  return new kibana.Plugin({
    require: ['elasticsearch'],
    name: 'comments',

    init: initServer,

    uiExports: {

      app: {
        title: 'Comments',
        description: '',
        main: 'plugins/comments/app',
        euiIconType: 'tag'
      }

    },

    config(Joi) {

      return Joi.object({
        enabled: Joi.boolean().default(true),
        newIndexNumberOfShards: Joi.number().integer().default(1),
        newIndexNumberOfReplicas: Joi.number().integer().default(1)
      }).default();
    },

    init(server, options) {

      // Add server routes and initialize the plugin here
      const dataCluster = server.plugins.elasticsearch.getCluster('data')

      serverRouteEsComment(server, dataCluster)
      serverRouteEsIndex(server, dataCluster)
    }


  });
};
