@description('Container image to deploy')
param containerImage string = 'ghcr.io/naltatis/evopt:latest'

@description('Azure region for all resources')
param location string = 'germanywestcentral'

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2025-02-01' = {
  name: 'evopt-logs'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

resource containerAppEnv 'Microsoft.App/managedEnvironments@2025-01-01' = {
  name: 'evopt-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

resource containerApp 'Microsoft.App/containerApps@2025-01-01' = {
  name: 'evopt'
  location: location
  properties: {
    managedEnvironmentId: containerAppEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 7050
      }
    }
    template: {
      containers: [
        {
          name: 'evopt'
          image: containerImage
          resources: {
            cpu: json('1')
            memory: '2Gi'
          }
          env: [
            { name: 'OPTIMIZER_TIME_LIMIT', value: '25' }
            { name: 'OPTIMIZER_NUM_THREADS', value: '1' }
            { name: 'GUNICORN_CMD_ARGS', value: '--workers 4 --max-requests 32' }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 3
        rules: [
          {
            name: 'http-scaling'
            http: {
              metadata: {
                concurrentRequests: '10'
              }
            }
          }
        ]
      }
    }
  }
}

output fqdn string = containerApp.properties.configuration.ingress.fqdn
