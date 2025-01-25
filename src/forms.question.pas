{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit forms.question;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, Graphics, Dialogs, StdCtrls;

type

  { TFormQuestion }

  TFormQuestion = class(TForm)
    ButtonCancel: TButton;
    ButtonInterrupt: TButton;
    procedure ButtonCancelKeyDown(Sender: TObject; var Key: Word;
      Shift: TShiftState);
    procedure ButtonInterruptKeyDown(Sender: TObject; var Key: Word;
      Shift: TShiftState);
  private

  public

  end;

var
  FormQuestion: TFormQuestion;

implementation

{$R *.lfm}

uses LCLType;

{ TFormQuestion }

procedure TFormQuestion.ButtonCancelKeyDown(Sender: TObject; var Key: Word;
  Shift: TShiftState);
begin
  case Key of
    VK_RIGHT, VK_RETURN : { do nothing };
    otherwise Key := 0;
  end;
end;

procedure TFormQuestion.ButtonInterruptKeyDown(Sender: TObject; var Key: Word;
  Shift: TShiftState);
begin
  case Key of
    VK_LEFT, VK_RETURN : { do nothing };
    otherwise Key := 0;
  end;
end;

end.

