#version 330 core
in vec3 ourColor;
in vec2 ourTexCoord;
uniform sampler2D ourTexture1;
uniform sampler2D ourTexture2;
out vec4 FragColor;

void main()
{
	int r = 5;
    vec2 texSize = vec2(380, 380);
	vec4 color = vec4(0,0,0,0);
    for (int y = -r; y <= r; y++) {
        for (int x = -r; x <= r; x++) {
				color += texture(ourTexture2, ourTexCoord + (vec2(x, y)/texSize));
        }
    }
    FragColor = color / ((2 * r + 1) * (2 * r + 1));
}