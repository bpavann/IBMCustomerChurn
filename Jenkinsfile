pipeline {
    agent any

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Docker Build') {
            steps {
                sh 'docker build -t ibmcustomerchurn:${BUILD_NUMBER} .'
            }
        }

        stage('Application Smoke Test') {
            steps {
                sh '''
                    docker run -d \
                        --name churn-test \
                        -p 5001:5000 \
                        ibmcustomerchurn:${BUILD_NUMBER}

                    sleep 10

                    docker exec churn-test \
                        python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:5000/').status)"
                '''
            }
        }
    }

    post {
        always {
            sh 'docker rm -f churn-test || true'
            sh 'docker rmi ibmcustomerchurn:${BUILD_NUMBER} || true'
        }
    }
}