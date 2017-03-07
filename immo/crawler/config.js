'use strict';

// Check configuration
checkConfig(process.env.COMMANDER_PASSWORD, 'COMMANDER_PASSWORD');

if (emptyConfig(process.env.PROVIDERS_TYPE)) {
    checkConfig(process.env.PROVIDERS_AWSEC2_ACCESSKEYID, 'PROVIDERS_AWSEC2_ACCESSKEYID');
    checkConfig(process.env.PROVIDERS_AWSEC2_SECRETACCESSKEY, 'PROVIDERS_AWSEC2_SECRETACCESSKEY');
}

// Export configuration
module.exports = {
    proxy: {
        port: parseInt(process.env.PROXY_PORT || '8888'),
    },

    commander: {
        port: parseInt(process.env.COMMANDER_PORT || '8889'),

        password: process.env.COMMANDER_PASSWORD,
    },

    instance: {
        checkDelay: parseInt(process.env.INSTANCE_CHECKDELAY || '10000'),
        checkAliveDelay: parseInt(process.env.INSTANCE_CHECKALIVEDELAY || '20000'),
        stopIfCrashedDelay: parseInt(process.env.INSTANCE_STOPIFCRASHEDDELAY || '300000'),
        addProxyNameInRequest: process.env.INSTANCE_ADDPROXYNAMEINREQUEST === 'false',

        autorestart: {
            minDelay: parseInt(process.env.INSTANCE_AUTORESTART_MINDELAY || '3600000'),
            maxDelay: parseInt(process.env.INSTANCE_AUTORESTART_MAXDELAY || '43200000'),
        },

        port: parseInt(process.env.INSTANCE_PORT || '3128'),

        scaling: {
            min: parseInt(process.env.INSTANCE_SCALING_MIN || '1'),
            max: parseInt(process.env.INSTANCE_SCALING_MAX || '2'),

            downscaleDelay: parseInt(process.env.INSTANCE_SCALING_DOWNSCALEDELAY || '600000'),
        },
    },

    providers: {
        type: process.env.PROVIDERS_TYPE || 'awsec2',
        awsec2: {
            accessKeyId: process.env.PROVIDERS_AWSEC2_ACCESSKEYID,
            secretAccessKey: process.env.PROVIDERS_AWSEC2_SECRETACCESSKEY,
            region: process.env.PROVIDERS_AWSEC2_REGION || 'eu-west-1',
            instance: {
                InstanceType: process.env.PROVIDERS_AWSEC2_INSTANCE_INSTANCETYPE || 't1.micro',
                ImageId: process.env.PROVIDERS_AWSEC2_INSTANCE_IMAGEID || 'ami-c74d0db4',
                SecurityGroups: [
                    process.env.PROVIDERS_AWSEC2_INSTANCE_SECURITYGROUPS || 'forward-proxy'
                ],
            },
            tag: process.env.PROVIDERS_AWSEC2_TAG || 'Proxy',
            maxRunningInstances: parseInt(process.env.PROVIDERS_AWSEC2_MAXRUNNINGINSTANCES || '10'),
        },

        ovhcloud: {
            endpoint: process.env.PROVIDERS_OVHCLOUD_ENDPOINT,
            appKey: process.env.PROVIDERS_OVHCLOUD_APPKEY,
            appSecret: process.env.PROVIDERS_OVHCLOUD_APPSECRET,
            consumerKey: process.env.PROVIDERS_OVHCLOUD_CONSUMERKEY,
            serviceId: process.env.PROVIDERS_OVHCLOUD_SERVICEID,
            region: process.env.PROVIDERS_OVHCLOUD_REGION,
            sshKeyName: process.env.PROVIDERS_OVHCLOUD_SSHKEYNAME,
            flavorName: process.env.PROVIDERS_OVHCLOUD_FLAVORNAME,
            snapshotName: process.env.PROVIDERS_OVHCLOUD_SNAPSHOTNAME,
            name: process.env.PROVIDERS_OVHCLOUD_NAME || 'Proxy',
            maxRunningInstances: parseInt(process.env.PROVIDERS_OVHCLOUD_MAXRUNNINGINSTANCES || '10'),
        },

        digitalocean: {
            token: process.env.PROVIDERS_DIGITALOCEAN_TOKEN,
            region: process.env.PROVIDERS_DIGITALOCEAN_REGION,
            size: process.env.PROVIDERS_DIGITALOCEAN_SIZE,
            sshKeyName: process.env.PROVIDERS_DIGITALOCEAN_SSHKEYNAME,
            imageName: process.env.PROVIDERS_DIGITALOCEAN_IMAGENAME,
            name: process.env.PROVIDERS_DIGITALOCEAN_NAME || 'Proxy',
            maxRunningInstances: parseInt(process.env.PROVIDERS_DIGITALOCEAN_MAXRUNNINGINSTANCES || '10'),
        },
    },

    stats: {
        retention: parseInt(process.env.STATS_RETENTION || '86400000'),

        samplingDelay: parseInt(process.env.STATS_SAMPLINGDELAY || '1000'),
    },
};


////////////

function emptyConfig(value) {
    return !value || value.length <= 0;
}

function checkConfig(value, name) {
    if (emptyConfig(value)) {
        throw new Error(`Cannot find environment variable ${name}`);
    }
}
