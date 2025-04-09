#!/bin/bash

echo "start supervisor"
supervisord -c supervisord.conf
