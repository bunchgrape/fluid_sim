#version 330 core
out vec4 FragColor;
in vec3 ourColor;

uniform float scale;

void main()
{
    FragColor = vec4(ourColor*scale, 1.0);
}