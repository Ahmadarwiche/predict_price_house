from django import forms




FLOOR_CHOICES = (
        (0.0, '0.0'),
        (0.5, '0.5'), 
        (1, '1'), 
        (1.5, '1.5'), 
        (2, '2'), 
        (2.5, '2.5'),
        (3, '3'),
        (3.5, '3.5'), 
        (4, '4')
    )


class FloorsForm(forms.Form):
    floors = forms.ChoiceField(label="Nombre d'étages", choices=FLOOR_CHOICES, error_messages={'required': "Veuillez indiquer le nombre d'étages"})