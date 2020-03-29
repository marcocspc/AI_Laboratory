# DeepRTS MACOS Dockerfile

Credits to perara for original code. You can see it on:

```

https://github.com/cair/deep-rts

```

## Description

This builds a DeepRTS image to be used on MacOS. 

## Important

- THIS WILL ONLY WORK IF YOU SET DOCKER CONTAINER's AVAILABLE MEMORY TO 3GB ON DOCKER's SETTINGS!
- You should install XQuartz to be able to see DeepRTS PyWindow:
    - After finishing installation, open xQuartz settings, go to Security and tick "Allow connections from network clients";
    - Then you should run this command on your terminal EVERY TIME YOU START xQuartz: `xhost + 127.0.0.1`; 
- You should mount the folder where you scripts are to /root/git;

Therefore, the 'docker run' command should be like this: 

```
    docker run -it \
           -v <your-script-dir>:/git/ \
           --name deeprts-gui-container
           deeprts-gui
```

By the way, the "docker build" command should be as follows:

```
    docker build . -t deeprts-gui
```

