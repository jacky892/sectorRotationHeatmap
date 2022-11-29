pipeline {
    agent any
    stages {
        stage('Checkout') {
            steps {
                checkout([$class: 'GitSCM', branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[url: 'https://github.com/jacky892/sectorRotationHeatmap.git']]])
            }
        }
        stage('Build') {
            steps {
                git branch: 'main', url: 'https://github.com/jacky892/sectorRotationHeatmap.git'
                sh 'pip install -r requirements.txt'
            }
        }
        stage('getdata') {
            steps {
                sh 'python3 update_data.py'
            }
        }
        stage('Test') {
            steps{ 
                dir("tests") {
                    sh 'pytest -s --log-cli-level=INFO'
                }
            }
        }
        
    }
}
