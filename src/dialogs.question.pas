{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit dialogs.question;

{$mode ObjFPC}{$H+}

interface

  function IsSessionCanceled : Boolean;

implementation

uses Controls, forms.question;

function IsSessionCanceled: Boolean;
begin
  FormQuestion := TFormQuestion.Create(nil);
  try
    case FormQuestion.ShowModal of
      mrCancel : begin
        Result := True;
      end;

      otherwise begin
        Result := False;
      end;
    end;
  finally
    FormQuestion.Free;
    FormQuestion := nil;
  end;
end;

end.

